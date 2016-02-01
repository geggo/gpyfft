# coding: utf-8
# TODO: create actual unit-tests
from __future__ import division, absolute_import
__doc__ = """Some helper function for tests"""
author = "Jérôme Kieffer"
__date__ = "21/01/2016"
__license__ = "LGPL"

import os
import unittest
import logging
import pyopencl as cl


logger = logging.getLogger(os.path.basename(__file__))


ALL_DEVICES = []
for platform in cl.get_platforms():
    ALL_DEVICES += platform.get_devices()


def get_contexts():
    return [ cl.Context([device]) for device in ALL_DEVICES ]


class TestOpenCL(unittest.TestCase):
    def test_context(self):
        """
        Create a context on every single
        """
        for ctx in get_contexts():
            logger.info("Platform: %s\t Device %s", ctx.devices[0].name, ctx.devices[0].platform.name)


def suite():
    """create test suite and returns it
    :return: test-suite
    """

    testsuite = unittest.TestSuite()
    testsuite.addTest(TestOpenCL("test_context"))
    return testsuite


def run():
    runner = unittest.TextTestRunner()
    runner.run(suite())

if __name__ == '__main__':
    run()
