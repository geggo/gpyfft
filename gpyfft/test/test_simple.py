# coding: utf-8
# TODO: create actual unit-tests
from __future__ import absolute_import, division, print_function

__doc__ = """Some tests"""
__author__ = "Jérôme Kieffer"
__date__ = "21/01/2016"  # coding: utf-8
__license__ = "LGPL"

import unittest
import logging
import time
import os
logger = logging.getLogger(os.path.basename(__file__))
import pyopencl as cl
import pyopencl.array as cla
import numpy as np

if __name__ == '__main__' and __package__ is None:
    __package__ = 'gpyfft.test'

from gpyfft.gpyfftlib import GpyFFT
from gpyfft.test.opencl import get_contexts


class TestSimple(unittest.TestCase):
    def test_simple(self):
        self.G = GpyFFT(debug=False)
        self.shape = (4, 8, 16)
        logger.info("clAmdFft Version: %s, \t Shape: %s", self.G.get_version(), self.shape)
        for ctx in get_contexts():
            self.simple(ctx)

    def simple(self, context):

        queue = cl.CommandQueue(context)

        logger.info("Platform: %s \t Device: %s", context.devices[0].name, context.devices[0].platform.name)
        logger.debug("context: 0x%x \t queue: 0x%x", context.int_ptr, queue.int_ptr)


        nd_data = np.random.random(self.shape).astype(np.complex64)
        cl_data = cla.to_device(queue, nd_data)
        cl_data_transformed = cla.empty_like(cl_data)

        logger.debug('nd_data.shape/strides %s / %s ' % (nd_data.shape, nd_data.strides))
        logger.debug('cl_data.shape/strides %s / %s' % (cl_data.shape, cl_data.strides))
        logger.debug('cl_data_transformed.shape/strides %s / %s ' % (cl_data_transformed.shape, cl_data_transformed.strides))

        plan = self.G.create_plan(context, cl_data.shape)
        plan.strides_in = tuple(x // cl_data.dtype.itemsize for x in cl_data.strides)
        plan.strides_out = tuple(x // cl_data.dtype.itemsize for x in cl_data_transformed.strides)

        logger.debug('plan.strides_in %s' % str(plan.strides_in))
        logger.debug('plan.strides_out %s' % str(plan.strides_out))
        logger.debug('plan.distances %s' % str(plan.distances))
        logger.debug('plan.batch_size %s' % str(plan.batch_size))

        logger.debug('plan.inplace: %s' % plan.inplace)
        plan.inplace = False
        logger.debug('plan.inplace: %s' % plan.inplace)

        logger.debug('plan.layouts: %s' % str(plan.layouts))

        plan.precision = 1
        logger.debug('plan.precision: %s' % plan.precision)

        plan.scale_forward = 10
        logger.debug('plan.scale_forward: %s' % plan.scale_forward)
        plan.scale_forward = 1
        logger.debug('plan.scale_forward: %s' % plan.scale_forward)

        logger.debug('plan.transpose_result: %s' % plan.transpose_result)

        t0 = time.time()
        plan.bake(queue)
        logger.debug("Bake time %.3fs", (time.time() - t0))
        logger.debug('plan.temp_array_size: %s', plan.temp_array_size)

        t1 = time.time()
        plan.enqueue_transform((queue,),
                               (cl_data.data,),
                               (cl_data_transformed.data,))
        queue.finish()
        t2 = time.time()

        t3 = time.time()
        res = np.fft.fftn(nd_data)
        t4 = time.time()
        error = abs(res - cl_data_transformed.get()).max()
        logger.info("Error: %s", error)
        logger.info("Numpy %.3fs; OpenCL %.3fs; Speed_up: %.3f" % (t4 - t3, t2 - t1, (t4 - t3) / (t2 - t1)))
        del plan
        self.assertLess(error, 4e-5, "Error %s <3e-5" % error)


def suite():
    """create test suite and returns it

    :return: test-suite
    """
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestSimple("test_simple"))
    return testsuite


def run():
    runner = unittest.TextTestRunner()
    runner.run(suite())

if __name__ == '__main__':
    run()


