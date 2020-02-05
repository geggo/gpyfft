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

def test3d(N=256, double_precision=False, padding=(0,0,0), axes_list=None):

    dtype = np.complex64 if not double_precision else np.complex128

    mem_shape = tuple((N + n_pad for n_pad in padding))

    array_slice = (slice(N), slice(N), slice(N))
    
    n_run = 10 #set to 1 for testing for correct result

    if n_run > 1:
        nd_dataC_raw = np.random.normal(size=mem_shape).astype(dtype)
        nd_dataC = nd_dataC_raw[array_slice]
    else:
        nd_dataC_raw = np.ones(mem_shape, dtype = dtype) #set n_run to 1
        nd_dataC = nd_dataC_raw[array_slice]

    nd_dataF_raw = np.asfortranarray(nd_dataC_raw)
    nd_dataF = nd_dataF_raw[array_slice]
    
    dataC_raw = cla.to_device(queue, nd_dataC_raw)
    dataF_raw = cla.to_device(queue, nd_dataF_raw)

    dataC = dataC_raw[array_slice]
    dataF = dataF_raw[array_slice]
    
    nd_resultC_raw = np.zeros_like(nd_dataC_raw, dtype = dtype)
    nd_resultF_raw = np.asfortranarray(nd_resultC_raw)
    
    resultC_raw = cla.to_device(queue, nd_resultC_raw)
    resultF_raw = cla.to_device(queue, nd_resultF_raw)
    resultC = resultC_raw[array_slice]
    resultF = resultF_raw[array_slice]
    
    result = resultF

    if axes_list is None:
        axes_list = [(-3,-2,-1), (-1,-2,-3), None] #batched 2d transforms

    if True:
        print('transform shape/padding', dataC.shape, dataC.dtype, padding)
        print('axes              in out')
        for axes in axes_list:
            for data in (dataC,
                         #dataF,
                        ):
                for result in (resultC,
                               None #inplace transform
                               #resultF,
                ):
                    t_ms, gflops = 0, 0
                    try:

                        transform = FFT(context, queue, data, result, axes = axes)
                        #transform.plan.transpose_result = True #not implemented for some transforms (works e.g. for out of place, (2,1) C C)
                        print('%-15s %3s %3s'
                               % (
                                   axes,
                                   'C' if data.flags.c_contiguous else 'F' if data.flags.f_contiguous else '?',
                                   '' if result is None else 'C' if result.flags.c_contiguous else 'F' if result.flags.f_contiguous else '?',
                               ),
                              end=' ',
                        )

                        durations = []
                        tic = timeit.default_timer()
                        
                        for i in range(n_run):
                            events = transform.enqueue()

                            for e in events:
                                e.wait()
                                #durations.append((e.profile.end-e.profile.start)*1e-6)
                                #durations.append((e.profile.end-e.profile.submit)*1e-6)
                                durations.append((e.profile.end-e.profile.queued)*1e-6)
                        min_duration_ms = min(durations)
                        #print("%.2f"%(min_duration_ms))
                                
                        toc = timeit.default_timer()
                        t_ms = 1e3*(toc-tic)/n_run
                        gflops = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*t_ms)
                        gflops_profiling = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*min_duration_ms)

                        if False: #check accuracy
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
                    except AssertionError as e:
                        print(e)
                    except Exception as e:
                        print(e)
                    finally:
                        print('%5.2fms %6.2f Gflops %6.2f Gflops peak' % (t_ms, gflops, gflops_profiling) )

                print()



if __name__ == '__main__':
    context = cl.create_some_context()
    queue = cl.CommandQueue(context,
                            properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    N = 256
    test3d(N, padding = (0,0,0), axes_list=[None])
    test3d(N, padding = (0,2,3), axes_list=[None])
    test3d(N, padding = (0,0,1), axes_list=[None])
    #test3d(N, padding = (0,0,2), axes_list=[None])
    #test3d(N, padding = (0,0,4), axes_list=[None])
    #test3d(N, padding = (0,0,8), axes_list=[None])
    #test3d(N, padding = (0,0,16), axes_list=[None])
    #test3d(N, padding = (0,0,32), axes_list=[None])
    #test3d(N, padding = (0,0,64), axes_list=[None])
    test3d(N, padding = (0,1,0), axes_list=[None])
    test3d(N, padding = (0,1,1), axes_list=[None])
    test3d(N, padding = (0,2,1), axes_list=[None])
    test3d(N, padding = (0,1,2), axes_list=[None])
    test3d(N, padding = (0,2,2), axes_list=[None])
    test3d(N, padding = (0,2,3), axes_list=[None])
    test3d(N, padding = (0,3,2), axes_list=[None])

    test3d(N, padding = (1,2,3), axes_list=[None])

    
    test3d(N, padding = (0,2,3), axes_list=[None], double_precision=True)
    #test3d(N=256, padding = (0,0,0), axes_list=[None])
    

