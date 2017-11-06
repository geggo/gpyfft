gpyfft
======

A Python wrapper for the OpenCL FFT library clFFT.

## Introduction

### clFFT

The open source library [clFFT] implements FFT for running on a GPU via OpenCL. Some highlights are:

* batched 1D, 2D, and 3D transforms
* supports many transform sizes (any combinatation of powers of 2,3,5,7,11, and 13)
* flexible memory layout
* single and double precisions
* complex and real-to-complex transforms
* supports injecting custom code for data pre- and post-processing

### gpyfft

This python wrapper is designed to tightly integrate with [PyOpenCL]. It consists of a low-level Cython based wrapper with an interface similar to the underlying C library. On top of that it offers a high-level interface designed to work on data contained in instances of `pyopencl.array.Array`, a numpy work-alike array class for GPU computations. The high-level interface takes some inspiration from [pyFFTW]. For details of the high-level interface see [fft.py].

## News
* 2017/11/05 for 2D and 3D transforms with default (empty) settings for the transform axes, now a more clever ordering of the transform axes is chosen, depending on the memory layout: last axis is transformed first for a C contiguous input array. I have seen huge performance improvements, 3x to 4x compared to the previous approach (always first axis first). Please report back benchmark results ('python -m gpyfft.benchmark') if this holds true for your GPU.

## Status

The low lever interface is complete (more or less), the high-level interface is not yet settled and likely to change in future. Features to come (not yet implemented in the high-level interface):

### work done

-   low level wrapper (mostly) completed
-   high level wrapper

  * complex-to-complex transform, in- and out-of-place
  * real-to-complex transform (out-of-place)
  * complex-to-real transform (out-of-place)
  * single precision
  * double precision 
  * interleaved data
  * support injecting custom OpenCL code (pre and post callbacks)
  * accept pyopencl arrays with non-zero offsets (Syam Gadde)
  * heuristics for optimal performance for choosing order axes transform if none given (Release 0.7.1)

## Basic usage

Here we describe a simple example of performing a batch of 2D complex-to-complex FFT transforms on the GPU, using the high-level interface of gpyfft. The full source code of this example ist contained in [simple\_example.py], which is the essence of [benchmark.py].
Note, for testing it is recommended to start [simple\_example.py] from the command line, so you have the possibility to interactively choose an OpenCL context (otherwise, e.g. when using an IPython, you are not asked end might end up with a CPU device, which is prone to fail). 

imports:

``` python
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT
```

initialize GPU:

``` python
context = cl.create_some_context()
queue = cl.CommandQueue(context)
```

initialize memory (on host and GPU). In this example we want to perform in parallel four 2D FFTs for 1024x1024 single precision data.

``` python
data_host = np.zeros((4, 1024, 1024), dtype = np.complex64)
#data_host[:] = some_useful_data
data_gpu = cla.to_device(queue, data_host)
```

create FFT transform plan for batched inline 2D transform along second two axes.

``` python
transform = FFT(context, queue, data_gpu, axes = (2, 1))
```

If you want an out-of-place transform, provide the output array as additional argument after the input data.

Start the work and wait until it is finished (Note that enqueu() returns a tuple of events)

``` python
event, = transform.enqueue()
event.wait()
```

Read back the data from the GPU to the host

``` python
result_host = data_gpu.get()
```

## Benchmark

A simple benchmark is contained as a submodule, you can run it on the command line by `python -m gpyfft.benchmark`, or from Python
``` python
import gpyfft.benchmark
gpyfft.benchmark.run()
```
Note, you might want to set the `PYOPENCL_CTX` environment variable to select your OpenCL platform and device.


  [clFFT]: https://github.com/clMathLibraries/clFFT
  [pyFFTW]: https://github.com/hgomersall/pyFFTW
  [PyOpenCL]: https://mathema.tician.de/software/pyopencl
  [fft.py]: gpyfft/fft.py
  [pyfft]: http://github.com/Manticore/pyfft
  [simple\_example.py]: examples/simple_example.py
  [benchmark.py]: gpyfft/benchmark.py
