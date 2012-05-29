gpyfft
======
A Python wrapper for the OpenCL FFT library APPML/clAmdFft from AMD

Introduction
------------
AMD has created a nice FFT library for use with their OpenCL implementation
called:

AMD Accelerated Parallel Processing Math Libraries<br/>
http://developer.amd.com/libraries/appmathlibs/Pages/default.aspx

But it doesn't generate a set of kernels and hand them back to the user to queue
up with the OpenCL kernel and event management system on his own.

Its programming model wants to control that part of the transaction, handing 
control back to the user only after the results are written to the output buffers.

So, if one wants to use this handy library with one's AMD setup and get the
benefit of PyOpenCL's simplified OpenCL interface this library wrapper is needed 
to make the AMD FFT library play nicely with PyOpenCL.

Requirements
------------
* python
* cython
* pyopencl
* APPML clAmdFft 1.8
* AMD APP SDK

Installation
------------

1)  Install the AMD library:

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
 GPL
 <br/>
(does it have to be a GPL?  Can we use one that's better for commercial and non-commercial use?)

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
    <td>?<p/></td>
    <td>AMD 6850</td>
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
