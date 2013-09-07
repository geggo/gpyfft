gpyfft
======

A Python wrapper for the OpenCL FFT library clFFT from AMD

Introduction
------------

AMD has created a nice FFT library for use with their OpenCL
implementation called `clFFT
<http://developer.amd.com/libraries/appmathlibs/Pages/default.aspx>`_
which has been released as open source (Apache 2.0 license)


This python wrapper is designed to tightly integrate with pyopencl. It
consists of a low-level cython based wrapper with an interface similar
to the underlying C library. On top of that it offers a high-level
interface designed to work on data contained in instances of
pyopencl.array.Array, a numpy work-alike array class. The high-level
interface is similar to that of `pyFFTW
<https://github.com/hgomersal/pyFFTW>`_, a python wrapper for the FFTW
library.

Compared to `pyfft <http://github.com/Manticore/pyfft>`_, a python
implementation of Apple's FFT library, AMD's FFT library offers some
additional features such as transform sizes that are powers of 2,3 and
5, and real-to-complex transforms. And on AMD hardware a better
performance can be expected, e.g., gpyfft: 280 Gflops compared to
pyfft: 63 GFlops (for single precision, accurate math,
inplace, transform size 1024x1024, batch size 4, on AMD Cayman, HD6950).


Status
------

This wrapper is currently under development.

work done
~~~~~~~~~

-  low level wrapper (mostly) completed
-  high level wrapper: complex (single precision), interleaved data, in
   and out of place (some tests and benchmarking available)
-  creation of pyopencl Events for synchronization

missing features
~~~~~~~~~~~~~~~~

-  debug mode to output generated kernels
-  documentation for low level wrapper (instead refer to library doc)
-  define API for high level interface
-  high level interface: double precision data, planar data,
   real<->complex transforms
-  high level interface: tests for non-contiguous data
-  handling of batched transforms in the general case, e.g. shape
   (4,5,6), axes = (1,), i.e., more than one axes where no transform is
   performed. (not always possible with single call for arbitrary
   strides, need to figure out when possible)

Requirements
------------

- python
- pyopencl (git version newer than 4 Jun 2012)
- distribute
- cython
- AMD clFFT

Building and Installation
-------------------------

1) Build and install the AMD clFFT library:

   - install clFFT
   - edit setup.py to point to clFFT directory

Then, either:

2) python setup.py install

or for developing

3) python setup.py develop 

   python setup.py build\_ext --inplace
   (rerun last command whenever cython extension has changed)

   python setup.py develop -u #for uninstalling development version

License:
--------

LGPL

Tested Platforms
----------------

+---------+-----------+-------+-----------------+----------------+----------+
|OS       |Python     |AMD APP|OpenCL           |Device          |Status    |
|         |           |       |                 |                |          |
|         |           |       |                 |                |          |
+---------+-----------+-------+-----------------+----------------+----------+
|Win7     |2.7, 64bit |2.7    |OpenCL 1.2,      |AMD Cayman      |works!    |
|(64bit)  |           |       |Catalyst 12.4    |(6950)          |          |
|         |           |       |                 |                |          |
+---------+-----------+-------+-----------------+----------------+----------+
|Win7     |2.7, 32bit |2.7    |OpenCL 1.1       |Intel i7        |works!    |
|(64bit)  |           |       |AMD-APP-SDK-v2.4 |                |          |
|         |           |       |(595.10)         |                |          |
+---------+-----------+-------+-----------------+----------------+----------+
|Win7     |2.7, 32bit |2.7    |OpenCL 1.1       |Intel i7        |works!    |
|(64bit)  |           |       |(Intel)          |                |          |
|         |           |       |                 |                |          |
+---------+-----------+-------+-----------------+----------------+----------+
|Win7     |2.7, 32bit |2.7    |OpenCL 1.0 CUDA  |Quadro 2000M    |Fails     |
|(64bit)  |           |       |4.0.1 (NVIDIA)   |                |          |
|         |           |       |                 |                |          |
+---------+-----------+-------+-----------------+----------------+----------+
|Win7     |2.7, 32bit |2.7    |OpenCL 1.2       |Tahiti (7970)   |works!    |
|(64bit)  |           |       |AMD-APP (923.1)  |                |          |
|         |           |       |                 |                |          |
+---------+-----------+-------+-----------------+----------------+----------+
|Win7     |2.7, 32bit |2.7    |OpenCL 1.2       |AMD Phenom IIx4 |works!    |
|(64bit)  |           |       |AMD-APP (923.1)  |                |          |
|         |           |       |                 |                |          |
+---------+-----------+-------+-----------------+----------------+----------+

(C) Gregor Thalhammer 2012

