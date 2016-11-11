from __future__ import absolute_import, division, print_function
import timeit
import numpy as np
from numpy.fft import fftn as npfftn
from numpy.testing import assert_array_almost_equal, assert_allclose
import pyopencl as cl
import pyopencl.array as cla
from gpyfft import FFT
from gpyfft.gpyfftlib import GpyFFT_Error

#real to complex: (forward) out_array.shape[axes][-1] = in_array.shape[axes][-1]//2 + 1

def run(double_precision=False):
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    dtype = np.complex64 if not double_precision else np.complex128
    
    n_run = 100 #set to 1 for proper testing

    Nx = 1024
    Ny = 1024
    batchsize = 4
    padx = 0
    pady = 0
    
    if n_run > 1:
        nd_dataCbase = np.random.normal(size=(batchsize,Ny+pady,Nx+padx)).astype(dtype)
        nd_dataC = nd_dataCbase[:batchsize, :Ny, :Nx]
    else:
        nd_dataC = np.ones((4,1024, 1024), dtype = dtype) #set n_run to 1

    dataCbase = cla.to_device(queue, nd_dataCbase)
    dataC = dataCbase[:batchsize, :Ny, :Nx]
    #dataC.set(nd_dataC)

    nd_dataFbase = np.asfortranarray(nd_dataCbase)
    nd_dataF = np.asfortranarray(nd_dataC) #TODO: also with padding (how?)
    dataF = cla.to_device(queue, nd_dataF)

    nd_result = np.zeros_like(nd_dataC, dtype = dtype)
    resultC = cla.to_device(queue, nd_result)
    resultF = cla.to_device(queue, np.asfortranarray(nd_result))
    result = resultF

    axes_list = [(1,2), (2,1)] #batched 2d transforms

    if True:
        print('out of place transforms', dataC.shape, dataC.dtype)
        print('axes         in out')
        for axes in axes_list:
            for layout_in, data in (('C', dataC),
                                    ('F', dataF)):
                for layout_out, result in (('C',resultC),
                                           ('F',resultF)):
                    try:

                        transform = FFT(context, queue, data, result, axes = axes)
                        #transform.plan.transpose_result = True #not implemented for some transforms (works e.g. for out of place, (2,1) C C)
                        print('%-10s %1s %1s'
                               % (
                                   axes,
                                   layout_in,  #'C' if data.flags.c_contiguous else 'F',
                                   layout_out, #'C' if result.flags.c_contiguous else 'F',
                               ),
                              end=' ',
                        )
                        
                        tic = timeit.default_timer()
                        for i in range(n_run):
                            events = transform.enqueue()
                            #events = transform.enqueue(False)
                        for e in events:
                            e.wait()
                        toc = timeit.default_timer()
                        t_ms = 1e3*(toc-tic)/n_run
                        gflops = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*t_ms)

                        npfft_result = npfftn(nd_dataC, axes = axes)
                        if transform.plan.transpose_result:
                            npfft_result = np.swapaxes(npfft_result, axes[0], axes[1])
                        max_error = np.max(abs(result.get() - npfft_result))
                        print('%8.1e'%max_error, end=' ')
                        assert_allclose(result.get(), npfft_result,
                                        atol = 1e-8 if double_precision else 1e-3,
                                        rtol = 1e-8 if double_precision else 1e-3)
                        
                        #assert_array_almost_equal(abs(result.get() - npfftn(data.get(), axes = axes)),
                        #                          1e-4)

   
                    except GpyFFT_Error as e:
                        print(e)
                        t_ms, gflops = 0, 0
                    except AssertionError as e:
                        print(e)
                    finally:
                        print('%5.2fms %6.2f Gflops' % (t_ms, gflops) )

        print('in place transforms', nd_dataC.shape, nd_dataC.dtype)

    for axes in axes_list:
        for nd_data in (nd_dataCbase, nd_dataFbase):
            data = cla.to_device(queue, nd_data)
            data = data[:batchsize, :Ny, :Nx]
            transform = FFT(context, queue, data, axes = axes)
            #transform.plan.transpose_result = True #not implemented
            tic = timeit.default_timer()
            for i in range(n_run):  # inplace transform fails for n_run > 1
                events = transform.enqueue()
            for e in events:
                    e.wait()
            toc = timeit.default_timer()
            t_ms = 1e3*(toc-tic)/n_run
            gflops = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*t_ms)
            print('%-10s %3s %5.2fms %6.2f Gflops' % (
                axes,
                'C' if data.flags.c_contiguous else 'F',
                t_ms, gflops
                ))
            #assert_array_almost_equal(data.get(queue=queue), npfftn(nd_data, axes = axes)) #never fails ????


if __name__ == '__main__':
    run()
    run(double_precision=True)
