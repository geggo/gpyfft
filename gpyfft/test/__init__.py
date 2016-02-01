# coding: utf-8
# TODO: create actual unit-tests
from __future__ import division, absolute_import
__doc__ = """Initialize test-suites"""
author = "Jérôme Kieffer"
__date__ = "21/01/2016"
__license__ = "LGPL"


import unittest
from . import opencl
from . import test_simple
from . import test_callback


def suite():
    """create test suite from all other suites
    :return: test-suite
    """
    testsuite = unittest.TestSuite()
    testsuite.addTest(opencl.suite())
    testsuite.addTest(test_simple.suite())
    testsuite.addTest(test_callback.suite())
    return testsuite


def run():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    run()
