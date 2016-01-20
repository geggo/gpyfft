from __future__ import absolute_import, division, print_function
import time
import numpy as np
from numpy.fft import fftn as npfftn
from numpy.testing import assert_array_almost_equal
import pyopencl as cl
import pyopencl.array as cla
from .fft import FFT
from .gpyfftlib import GpyFFT_Error

#complex transform: 2x input_arrays real or 1x input_array, same output
#real to complex: (forward) out_array.shape[axes][-1] = in_array.shape[axes][-1]//2 + 1

def run():


    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    n_run = 10 #set to 1 for proper testing

    if n_run > 1:
        #nd_dataC = np.zeros((1024, 1024), dtype = np.complex64) #for benchmark
        nd_dataC = np.zeros((4,1024, 1024), dtype = np.complex64) #for benchmark
        #nd_dataC = np.zeros((128,128,128), dtype = np.complex64) #for benchmark
    else:
        nd_dataC = np.ones((4,1024, 1024), dtype = np.complex64) #set n_run to 1

    #nd_dataC = np.array([[1,2,3,4], [5,6,7,8]], dtype = np.complex64) #small array

    nd_dataF = np.asfortranarray(nd_dataC)
    dataC = cla.to_device(queue, nd_dataC)
    dataF = cla.to_device(queue, nd_dataF)

    nd_result = np.zeros_like(nd_dataC, dtype = np.complex64)
    resultC = cla.to_device(queue, nd_result)
    resultF = cla.to_device(queue, np.asfortranarray(nd_result))
    result = resultF


    #axes_list = [(0,), (1,), (0,1)] #is (1,0) the same?
    #axes_list = [(1,0), (0,1), (1,2), (2,1)]
    #axes_list = [(1,2), (2,1)]
    axes_list = [(1,0), (0,1), (1,2), (2,1), (0,1,2), (2,1,0)]

    if True:
        print('out of place transforms', dataC.shape)
        print('axes         in out')
        for axes in axes_list:
            for data in (dataC, dataF):
                for result in (resultC, resultF):
                    try:

                        transform = FFT(context, queue, (data,), (result,), axes = axes)
                        #transform.plan.transpose_result = True #not implemented for some transforms (works e.g. for out of place, (2,1) C C)
                        tic = time.clock()
                        for i in range(n_run):
                            events = transform.enqueue()
                            #events = transform.enqueue(False)
                        for e in events:
                            e.wait()
                        toc = time.clock()
                        t_ms = 1e3*(toc-tic)/n_run
                        gflops = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*t_ms)
                        print('%-10s %3s %3s %5.2fms %6.2f Gflops' % (
                            axes,
                            'C' if data.flags.c_contiguous else 'F',
                            'C' if result.flags.c_contiguous else 'F',
                            t_ms, gflops
                            ))
                        assert_array_almost_equal(result.get(), npfftn(data.get(), axes = axes))
                    except GpyFFT_Error as e:
                        print(e)
                    except AssertionError as e:
                        print(e)

        print()
        print('in place transforms', nd_dataC.shape)

    for axes in axes_list:
        for nd_data in (nd_dataC, nd_dataF):
            data = cla.to_device(queue, nd_data)
            transform = FFT(context, queue, (data,), axes = axes)
            #transform.plan.transpose_result = True #not implemented
            tic = time.clock()
            for i in range(n_run):  # inplace transform fails for n_run > 1
                events = transform.enqueue()
            for e in events:
                    e.wait()
            toc = time.clock()
            t_ms = 1e3*(toc-tic)/n_run
            gflops = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*t_ms)
            print('%-10s %3s %5.2fms %6.2f Gflops' % (
                axes,
                'C' if data.flags.c_contiguous else 'F',
                t_ms, gflops
                ))
            #assert_array_almost_equal(data.get(queue=queue), npfftn(nd_data, axes = axes)) #never fails ????



    # #################
    # #check for block-contiguous -> axis can be collapsed

    # b = arange(24)
    # b.shape = (2,3,4)

    # #transform along last axis
    # a[:,:,0] #indices of individual transform subarrays
    # a[:,0,:]

    # #nditer, nested_iters,
