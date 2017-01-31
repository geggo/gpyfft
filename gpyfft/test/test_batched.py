from __future__ import print_function
import unittest
from nose_parameterized import parameterized
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft import FFT
from gpyfft.test.util import get_contexts

contexts = [(ctx,) for ctx in get_contexts()]

class test_fft_batched(unittest.TestCase):

    @parameterized.expand(contexts)
    def test_2d_out_of_place(self, ctx):
        queue = cl.CommandQueue(ctx)

        L = 4
        M = 64
        N = 32
        axes = (-1, -2)
        
        nd_data = np.arange(L*M*N, dtype=np.complex64)
        nd_data.shape = (L, M, N)
        cl_data = cla.to_device(queue, nd_data)
        
        cl_data_transformed = cla.zeros_like(cl_data)
        
        transform = FFT(ctx, queue,
                        cl_data,
                        cl_data_transformed,
                        axes = axes,
                        )

        transform.enqueue()

        print(cl_data_transformed.get)
        print(np.fft.fft2(nd_data))
        
        assert np.allclose(cl_data_transformed.get(),
                           np.fft.fft2(nd_data, axes=axes),
                           rtol=1e-3, atol=1e-3)


    @parameterized.expand(contexts)
    def test_2d_in_4d_out_of_place(self, ctx):
        queue = cl.CommandQueue(ctx)

        L1 = 4
        L2 = 5
        
        M = 64
        N = 32
        axes = (-1, -2) #ok
        #axes = (0,1) #ok
        #axes = (0,2) #cannot be collapsed
        
        nd_data = np.arange(L1*L2*M*N, dtype=np.complex64)
        nd_data.shape = (L1, L2, M, N)
        cl_data = cla.to_device(queue, nd_data)
        
        cl_data_transformed = cla.zeros_like(cl_data)
        
        transform = FFT(ctx, queue,
                        cl_data,
                        cl_data_transformed,
                        axes = axes,
                        )

        transform.enqueue()

        print(cl_data_transformed.get)
        print(np.fft.fft2(nd_data))
        
        assert np.allclose(cl_data_transformed.get(),
                           np.fft.fft2(nd_data, axes=axes),
                           rtol=1e-3, atol=1e-3)
