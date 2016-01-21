# coding: utf-8
from __future__ import absolute_import, division, print_function

__doc__ = """Some tests"""
__author__ = "Jérôme Kieffer"
__date__ = "21/01/2016"  # coding: utf-8
__license__ = "LGPL"

import unittest
import logging
import os
logger = logging.getLogger(os.path.basename(__file__))

import pyopencl as cl
import pyopencl.array as cla
import numpy as np
from ..gpyfftlib import GpyFFT
from .opencl import get_contexts


class TestCallback(unittest.TestCase):
    callback_kernel = """
float2 mulval(__global void* in,
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

    def test_callback(self):
        self.G = GpyFFT(debug=True)
        self.nd_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.complex64)

        logger.info("clAmdFft Version: %s, \t Shape: %s", self.G.get_version(), self.nd_data.shape)
        for ctx in get_contexts():
            self.simple(ctx)

    def simple(self, context):

        queue = cl.CommandQueue(context)

        logger.debug("context:", hex(context.int_ptr))
        logger.debug("queue:", hex(queue.int_ptr))
        nd_data = self.nd_data
        cl_data = cla.to_device(queue, nd_data)
        cl_data_transformed = cla.empty_like(cl_data)

        logger.debug("cl_data:")
        logger.debug(cl_data)

        logger.debug('nd_data.shape/strides', nd_data.shape, nd_data.strides)
        logger.debug('cl_data.shape/strides', cl_data.shape, cl_data.strides)
        logger.debug('cl_data_transformed.shape/strides', cl_data_transformed.shape, cl_data_transformed.strides)

        plan = self.G.create_plan(context, cl_data.shape)
        plan.strides_in = tuple(x // cl_data.dtype.itemsize for x in cl_data.strides)
        plan.strides_out = tuple(x // cl_data.dtype.itemsize for x in cl_data_transformed.strides)


        logger.debug('plan.strides_in', plan.strides_in)
        logger.debug('plan.strides_out', plan.strides_out)
        logger.debug('plan.distances', plan.distances)
        logger.debug('plan.batch_size', plan.batch_size)

        logger.debug('plan.inplace:', plan.inplace)
        plan.inplace = False
        logger.debug('plan.inplace:', plan.inplace)

        logger.debug('plan.layouts:', plan.layouts)

        plan.precision = 1
        logger.debug('plan.precision:', plan.precision)

        plan.scale_forward = 10
        logger.debug('plan.scale_forward: %s', plan.scale_forward)
        plan.scale_forward = 1
        logger.debug('plan.scale_forward: %s', plan.scale_forward)

        logger.debug('plan.transpose_result: %s', plan.transpose_result)


        user_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        user_data_device = cla.to_device(queue, user_data)

        plan.set_callback('mulval', self.callback_kernel, 'pre',
                               user_data=user_data_device.data)

        plan.bake(queue)
        logger.debug('plan.temp_array_size:', plan.temp_array_size)

        plan.enqueue_transform((queue,),
                               (cl_data.data,),
                               (cl_data_transformed.data,)
                               )
        queue.finish()

        logger.debug('cl_data:')
        logger.debug(cl_data)
        logger.debug('cl_data_transformed:')
        logger.debug(cl_data_transformed)


        logger.debug('nd_data:')
        logger.debug(nd_data)
        logger.debug('fft(nd_data):')
        logger.debug(np.fft.fftn(nd_data))

        del plan


def suite():
    """create test suite and returns it

    :return: test-suite
    """
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestCallback("test_callback"))
    return testsuite


def run():
    runner = unittest.TextTestRunner()
    runner.run(suite())

if __name__ == '__main__':
    run()


