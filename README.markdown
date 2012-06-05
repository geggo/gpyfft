gpyfft
======
A Python wrapper for the OpenCL FFT library APPML/clAmdFft from AMD

Introduction
------------
AMD has created a nice FFT library for use with their OpenCL implementation
called:

AMD Accelerated Parallel Processing Math Libraries<br/>
http://developer.amd.com/libraries/appmathlibs/Pages/default.aspx

This C library is available as precompiled binaries for Windows and
Linux platforms. 

This python wrapper is designed to tightly integrate with pyopencl. It
consists of a low-level cython based wrapper with an interface similar
to the underlying C library. On top of that it offers a high-level
interface designed to work on data contained in instances of
pyopencl.array.Array, a numpy work-alike array class. The high-level
interface is similar to that of
[pyFFTW](https://github.com/hgomersal/pyFFTW), a python wrapper for
the FFTW library.

Status
------
This wrapper is currently under development.

### work done ###
* low level wrapper (mostly) completed
* high level wrapper: complex (single precision), interleaved data, in
  and out of place (some tests and benchmarking available)
* creation of pyopencl Events for synchronization
  
### missing features ###
* debug mode to output generated kernels
* documentation for low level wrapper (instead refer to library doc)
* define API for high level interface
* high level interface: double precision data, planar data, real<->complex transforms
* high level interface: tests for non-contiguous data
* handling of batched transforms in the general case, e.g. shape
  (4,5,6), axes = (1,), i.e., more than one axes where no transform is
  performed. (not always possible with single call for arbitrary
  strides, need to figure out when possible)

Requirements
------------
* python
* pyopencl (git version newer than 4 Jun 2012)
* cython
* APPML clAmdFft 1.8
* AMD APP SDK

Installation
------------
1) Install the AMD library:

   * install clAmdFft
   * add clAmdFft/binXX to PATH, or copy clAmdFft.Runtime.dll to package directory
   * edit setup.py to point to clAmdFft and AMD APP directories

  Then, either:
  
2a) python setup.py install
  <p/>Or:<p/>
2b) inplace build:
  python setup.py build_ext --inplace

License:
--------
LGPL
 
Tested Platforms
----------------
<table width="100%">
  <tr>
    <th>OS</th><th>Python</th><th>AMD APP</th><th>OpenCL</th><th>Device</th><th>Status</th>
  </tr>
  <tr>
    <td>Win7 (64 bit)</td>
    <td>2.7 (64bit)</td>
    <td>2.7</td>
    <td>OpenCL 1.2<br/>Catalyst 12.4</td>
    <td>Cayman (6850)</td>
    <td>works!</td>
  </tr>
  
  <tr>
    <td>Win7 (64 bit)</td>
    <td>2.7 (32bit)</td>
    <td>2.7</td>
    <td>OpenCL 1.1<br/>AMD-APP-SDK-v2.4<br/>(595.10)<p/></td>
    <td>Intel i7</td>
    <td>works!</td>    
  </tr>
  
  <tr>
    <td>Win7 (64 bit)</td>
    <td>2.7 (32bit)</td>
    <td>2.7</td>
    <td>OpenCL 1.1<br/>(Intel)<p/></td>
    <td>Intel i7</td>
    <td>works!</td>    
  </tr>

  <tr>
    <td>Win7 (64 bit)</td>
    <td>2.7 (32bit)</td>
    <td>2.7</td>
    <td>OpenCL 1.0<br/>CUDA 4.0.1<br/>(NVIDIA)<p/></td>
    <td>Quadro 2000M</td>
    <td>Fails</td>    
  </tr>
  
  <tr>
    <td>Win7 (64 bit)</td>
    <td>2.7 (32bit)</td>
    <td>2.7</td>
    <td>OpenCL 1.2<br/>AMD-APP<br/>(923.1)<p/></td>
    <td>Tahiti (7970)</td>
    <td>works!</td>    
  </tr>

  <tr>
    <td>Win7 (64 bit)</td>
    <td>2.7 (32bit)</td>
    <td>2.7</td>
    <td>OpenCL 1.2<br/>AMD-APP<br/>(923.1)<p/></td>
    <td>AMD Phenom IIx4</td>
    <td>works!</td>    
  </tr>

  <tr>
    <td>Win7 (64 bit)</td>
    <td>2.7 (32bit)</td>
    <td>2.7</td>
    <td>OpenCL 1.2<br/>AMD-APP<br/>(923.1)<p/></td>
    <td>HD7970<br/>(Tahiti 925MHz)</td>
    <td>works!</td>    
  </tr>

</table>

(C) Gregor Thalhammer 2012
