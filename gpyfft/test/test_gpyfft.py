from __future__ import print_function
import unittest
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft import FFT
from gpyfft.test.util import get_contexts


"""
Some basic tests for high-level interface
"""

class test_fft(unittest.TestCase):

    def test_1d_inplace(self):
        ctx = get_contexts()[0]
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
        

    def test_1d_out_of_place(self):
        ctx = get_contexts()[0]
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


if __name__ == '__main__':
    unittest.main()
