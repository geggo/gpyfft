# coding: utf-8
from __future__ import absolute_import, division, print_function

import unittest
import os

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

from gpyfft.gpyfftlib import *
from gpyfft.test.util import get_contexts


class TestCallbackPreMul(unittest.TestCase):

    
    callback_kernel_src_premul = b"""
float2 premul(__global void* in,
              uint inoffset,
              __global void* userdata
              //__local void* localmem
)
{
float scalar = *((__global float*)userdata + inoffset);
float2 ret = *((__global float2*)in + inoffset) * scalar;
return ret;
}
"""

    def test_callback_pre(self):
        for ctx in get_contexts():
            self.callback_pre(ctx)

    def callback_pre(self, context):
        print("context:", context)
        queue = cl.CommandQueue(context)

        nd_data = np.array([[1, 2, 3, 4],
                            [5, 6, 5, 2]],
                           dtype=np.complex64)
        cl_data = cla.to_device(queue, nd_data)
        cl_data_transformed = cla.empty_like(cl_data)

        print("cl_data:")
        print(cl_data)
        print('nd_data.shape/strides:', nd_data.shape, nd_data.strides)
        print('cl_data.shape/strides:', cl_data.shape, cl_data.strides)
        print('cl_data_transformed.shape/strides:', cl_data_transformed.shape, cl_data_transformed.strides)

        G = GpyFFT(debug=False)
        plan = G.create_plan(context, cl_data.shape)
        
        plan.strides_in  = tuple(x // cl_data.dtype.itemsize for x in cl_data.strides)
        plan.strides_out = tuple(x // cl_data.dtype.itemsize for x in cl_data_transformed.strides)
        print('plan.strides_in', plan.strides_in)
        print('plan.strides_out', plan.strides_out)
        print('plan.distances', plan.distances)
        print('plan.batch_size', plan.batch_size)

        plan.inplace = False
        
        plan.precision = CLFFT_SINGLE
        print('plan.precision:', plan.precision)

        plan.scale_forward = 1.
        print('plan.scale_forward:', plan.scale_forward)
        
        #print('plan.transpose_result:', plan.transpose_result)

        nd_user_data = np.array([[2, 2, 2, 2],
                                 [3, 4, 5, 6]],
                                dtype=np.float32)
        cl_user_data = cla.to_device(queue, nd_user_data)
        print('cl_user_data')
        print(cl_user_data)

        plan.set_callback(b'premul',
                          self.callback_kernel_src_premul,
                          'pre',
                          user_data=cl_user_data.data)

        plan.bake(queue)
        print('plan.temp_array_size:', plan.temp_array_size)

        plan.enqueue_transform((queue,),
                               (cl_data.data,),
                               (cl_data_transformed.data,)
                               )
        queue.finish()

        print('cl_data_transformed:')
        print(cl_data_transformed)

        print('fft(nd_data * nd_user_data):')
        print(np.fft.fftn(nd_data * nd_user_data))

        assert np.allclose(cl_data_transformed.get(),
                           np.fft.fftn(nd_data * nd_user_data))
        
        del plan


    callback_kernel_src_postset = b"""
    float2 postset(__global void* output,
                   uint offset,
                   __global void* userdata,
                   float2 fftoutput)
    {
       float scalar = *((__global float*)userdata + offset); 
       *((__global float2*)output + offset) = fftoutput * scalar;
    }
"""

    def test_callback_post(self):
        for ctx in get_contexts():
            self.callback_post(ctx)

    def callback_post(self, context):
        print("context:", context)
        queue = cl.CommandQueue(context)

        nd_data = np.array([[1, 2, 3, 4],
                            [5, 6, 5, 2]],
                           dtype=np.complex64)
        nd_user_data = np.array([[2, 2, 2, 2],
                                 [3, 4, 5, 6]],
                                dtype=np.float32)

        cl_data = cla.to_device(queue, nd_data)
        cl_user_data = cla.to_device(queue, nd_user_data)
        cl_data_transformed = cla.empty_like(cl_data)

        G = GpyFFT(debug=False)
        plan = G.create_plan(context, cl_data.shape)
        plan.strides_in  = tuple(x // cl_data.dtype.itemsize for x in cl_data.strides)
        plan.strides_out = tuple(x // cl_data.dtype.itemsize for x in cl_data_transformed.strides)
        plan.inplace = False
        plan.precision = CLFFT_SINGLE
        plan.set_callback(b'postset',
                          self.callback_kernel_src_postset,
                          'post',
                          user_data=cl_user_data.data)

        plan.bake(queue)
        plan.enqueue_transform((queue,),
                               (cl_data.data,),
                               (cl_data_transformed.data,)
                               )
        queue.finish()

        print('cl_data_transformed:')
        print(cl_data_transformed)

        print('fft(nd_data) * nd_user_data')
        print(np.fft.fftn(nd_data))

        assert np.allclose(cl_data_transformed.get(),
                           np.fft.fftn(nd_data) * nd_user_data)
        
        del plan
    

#TODO: create TestSuite
