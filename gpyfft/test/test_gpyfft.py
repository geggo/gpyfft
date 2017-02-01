from __future__ import print_function
import unittest
from nose_parameterized import parameterized
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft import FFT
from gpyfft.test.util import get_contexts, has_double


"""
Some basic tests for high-level interface
"""

#TODO: perform tests for all contexts

contexts = [(ctx,) for ctx in get_contexts()]

class test_fft(unittest.TestCase):

    @parameterized.expand(contexts)
    def test_1d_inplace(self, ctx):
        queue = cl.CommandQueue(ctx)
        
        nd_data = np.arange(32, dtype=np.complex64)
        cl_data = cla.to_device(queue, nd_data)
        #cl_data_transformed = cla.zeros_like(cl_data)
        
        transform = FFT(ctx, queue,
                        cl_data)
        transform.enqueue()
        #print(cl_data)
        #print(np.fft.fft(nd_data))

        assert np.allclose(cl_data.get(),
                           np.fft.fft(nd_data))

    @parameterized.expand(contexts)
    def test_1d_out_of_place(self, ctx):
        queue = cl.CommandQueue(ctx)
        
        nd_data = np.arange(32, dtype=np.complex64)
        cl_data = cla.to_device(queue, nd_data)
        cl_data_transformed = cla.zeros_like(cl_data)
        
        transform = FFT(ctx, queue,
                        cl_data,
                        cl_data_transformed
        )
        transform.enqueue()

        assert np.allclose(cl_data_transformed.get(),
                           np.fft.fft(nd_data))


    @parameterized.expand(contexts)
    def test_1d_inplace_double(self, ctx):
        if not has_double(ctx): #TODO: find better way to skip test
            return
        queue = cl.CommandQueue(ctx)
        
        nd_data = np.arange(32, dtype=np.complex128)
        cl_data = cla.to_device(queue, nd_data)
        
        transform = FFT(ctx, queue,
                        cl_data)
        transform.enqueue()

        assert np.allclose(cl_data.get(),
                           np.fft.fft(nd_data))

    @parameterized.expand(contexts)
    def test_1d_real_to_complex(self, ctx):
        queue = cl.CommandQueue(ctx)
        
        N = 32

        nd_data = np.arange(N, dtype=np.float32)
        cl_data = cla.to_device(queue, nd_data)
        cl_data_transformed = cla.zeros(queue, (N//2+1,), dtype = np.complex64)
        
        transform = FFT(ctx, queue,
                        cl_data,
                        cl_data_transformed,
        )
        transform.enqueue()

        assert np.allclose(cl_data_transformed.get(),
                           np.fft.rfft(nd_data))

    @parameterized.expand(contexts)
    def test_2d_real_to_complex(self, ctx):
        queue = cl.CommandQueue(ctx)
        
        M = 64
        N = 32

        nd_data = np.arange(M*N, dtype=np.float32)
        nd_data.shape = (M, N)
        cl_data = cla.to_device(queue, nd_data)
        
        cl_data_transformed = cla.zeros(queue, (M, N//2+1), dtype = np.complex64)
        
        transform = FFT(ctx, queue,
                        cl_data,
                        cl_data_transformed,
                        axes = (1,0),
                        )

        transform.enqueue()

        print(cl_data_transformed.get)
        print(np.fft.rfft2(nd_data))
        
        assert np.allclose(cl_data_transformed.get(),
                           np.fft.rfft2(nd_data),
                           rtol=1e-3, atol=1e-3)

    @parameterized.expand(contexts)
    def test_2d_real_to_complex_double(self, ctx):
        if not has_double(ctx): #TODO: find better way to skip test
            return
        queue = cl.CommandQueue(ctx)
        
        M = 64
        N = 32

        nd_data = np.arange(M*N, dtype=np.float64)
        nd_data.shape = (M, N)
        cl_data = cla.to_device(queue, nd_data)
        
        cl_data_transformed = cla.zeros(queue, (M, N//2+1), dtype = np.complex128)
        
        transform = FFT(ctx, queue,
                        cl_data,
                        cl_data_transformed,
                        axes = (1,0),
                        )

        transform.enqueue()

        print(cl_data_transformed.get)
        print(np.fft.rfft2(nd_data))
        
        assert np.allclose(cl_data_transformed.get(),
                           np.fft.rfft2(nd_data),
                           rtol=1e-8, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
