import gpyfft
GFFT = gpyfft.GpyFFT()
import time
import numpy as np

# TODO:
# real to complex: out-of-place
# planar, interleaved arrays
# precision: single, double

class FFT(object):
    def __init__(self, context, queue, input_arrays, output_arrays=None, axes = None, fast_math = False):
        self.context = context
        self.queue = queue

        in_array = input_arrays[0]
        t_strides_in, t_distance_in, t_batchsize_in, t_shape = self.calculate_transform_strides(
            axes, in_array.shape, in_array.strides, in_array.dtype)

        if output_arrays is not None:
            t_inplace = False
            out_array = output_arrays[0]
            t_strides_out, t_distance_out, foo, bar = self.calculate_transform_strides(
                axes, out_array.shape, out_array.strides, out_array.dtype)
        else:
            t_inplace = True
            out_array = None
            t_strides_out, t_distance_out = t_strides_in, t_distance_in

        self.t_shape = t_shape
        self.batchsize = t_batchsize_in

        plan = GFFT.create_plan(context, t_shape)
        plan.inplace = t_inplace
        plan.strides_in = t_strides_in
        plan.strides_out = t_strides_out
        plan.distances = (t_distance_in, t_distance_out)
        plan.batch_size = t_batchsize_in #assert t_batchsize_in == t_batchsize_out
        
        if False:
            print 'axes', axes        
            print 'in_array.shape:          ', in_array.shape
            print 'in_array.strides/itemsize', tuple(s/in_array.dtype.itemsize for s in in_array.strides)
            print 'shape transform          ', t_shape
            print 't_strides                ', t_strides_in
            print 'distance_in              ', t_distance_in
            print 'batchsize                ', t_batchsize_in
            print 't_stride_out             ', t_strides_out

        plan.bake(self.queue)
        temp_size = plan.temp_array_size
        if temp_size:
            #print 'temp_size:', plan.temp_array_size
            self.temp_buffer = cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size = temp_size)
        else:
            self.temp_buffer = None

        self.plan = plan
        self.data = in_array #TODO: planar arrays
        self.result = out_array #TODO: planar arrays


    def calculate_transform_strides(self,
                                    axes,
                                    shape, 
                                    strides,
                                    dtype,
                                   ):
        ddim = len(shape) #dimensionality data
        if axes is None:
            axes = range(ddim)

        tdim = len(axes) #dimensionality transform
        assert tdim <= ddim
        
        axes_transform = tuple(a + ddim if a<0 else a for a in axes)

        axes_notransform = set(range(ddim)).difference(axes_transform)
        assert len(axes_notransform) < 2, 'more than one non-transformed axis'
        #TODO: collapse non-transformed axes if possible

        t_shape = [shape[i] for i in axes_transform]
        dsize = dtype.itemsize
        t_strides = [strides[i]//dsize for i in axes_transform]
        
        t_distance = [strides[i]//dsize for i in axes_notransform]
        if not t_distance:
            t_distance = 0
        else:
            t_distance = t_distance[0] #TODO
    
        batchsize = 1
        for a in axes_notransform:
            batchsize *= shape[a]

        return (tuple(t_strides), t_distance, batchsize, tuple(t_shape))

    def enqueue(self, forward = True):
        """enqueue transform"""
        if self.result is not None:
            events = self.plan.enqueue_transform((self.queue,), (self.data.data,), (self.result.data),
                                        direction_forward = forward, temp_buffer = self.temp_buffer)
        else:
            events = self.plan.enqueue_transform((self.queue,), (self.data.data,),
                                        direction_forward = forward, temp_buffer = self.temp_buffer)
        return events

    def update_arrays(input_array, output_array):
        pass

#complex transform: 2x input_arrays real or 1x input_array, same output
#real to complex: (forward) out_array.shape[axes][-1] = in_array.shape[axes][-1]//2 + 1

import numpy as np
from numpy.fft import fftn as npfftn
from numpy.testing import assert_array_almost_equal
import pyopencl as cl
import pyopencl.array as cla

context = cl.create_some_context()
queue = cl.CommandQueue(context)

n_run = 10 #set to 1 for proper testing

if n_run > 1:
    #nd_dataC = np.zeros((1024, 1024), dtype = np.complex64) #for benchmark
    nd_dataC = np.zeros((4,1024, 1024), dtype = np.complex64) #for benchmark
    #nd_dataC = np.zeros((128,128,128), dtype = np.complex64) #for benchmark
else:
    nd_dataC = np.ones((1024, 1024), dtype = np.complex64) #set n_run to 1

#nd_dataC = np.array([[1,2,3,4], [5,6,7,8]], dtype = np.complex64) #small array

nd_dataF = np.asfortranarray(nd_dataC)
dataC = cla.to_device(queue, nd_dataC)
dataF = cla.to_device(queue, nd_dataF)

nd_result = np.zeros_like(nd_dataC, dtype = np.complex64)
resultC = cla.to_device(queue, nd_result)
resultF = cla.to_device(queue, np.asfortranarray(nd_result))
result = resultF


axes_list = [(0,), (1,), (0,1)] #is (1,0) the same?
axes_list = [(1,0), (0,1), (1,2), (2,1)]
#axes_list = [(1,0), (0,1), (1,2), (2,1), (0,1,2), (2,1,0)]

print 'out of place transforms', dataC.shape
print 'axes         in out'
for axes in axes_list:
    for data in (dataC, dataF):
        for result in (resultC, resultF):
    
            transform = FFT(context, queue, (data,), (result,), axes = axes)
            tic = time.clock()
            for i in range(n_run):
                events = transform.enqueue()
            for e in events:
                e.wait()
            toc = time.clock()
            t_ms = 1e3*(toc-tic)/n_run
            gflops = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*t_ms)
            print '%-10s %3s %3s %5.2fms %4d Gflops'%(
                axes,
                'C' if data.flags.c_contiguous else 'F',  
                'C' if result.flags.c_contiguous else 'F',  
                t_ms, gflops
                )
            assert_array_almost_equal(result.get(), npfftn(data.get(), axes = axes))

print
print 'in place transforms', nd_dataC.shape

for axes in axes_list:
    for nd_data in (nd_dataC, nd_dataF):
        data = cla.to_device(queue, nd_data)
        transform = FFT(context, queue, (data,), axes = axes)
        tic = time.clock()
        for i in range(n_run//2):
            events = transform.enqueue()
        for e in events:
                e.wait()
           
        toc = time.clock()
        t_ms = 1e3*(toc-tic)/n_run
        gflops = 5e-9 * np.log2(np.prod(transform.t_shape))*np.prod(transform.t_shape) * transform.batchsize / (1e-3*t_ms)
        print '%-10s %3s %5.2fms %4d Gflops'%(
            axes,
            'C' if data.flags.c_contiguous else 'F',
            t_ms, gflops
            )
        assert_array_almost_equal(data.get(), npfftn(nd_data, axes = axes))



# #################
# #check for block-contiguous -> axis can be collapsed

# b = arange(24)
# b.shape = (2,3,4)

# #transform along last axis
# a[:,:,0] #indices of individual transform subarrays
# a[:,0,:]

# #nditer, nested_iters, 
