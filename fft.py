from numpy.fft import fftn as npfftn
import gpyfft
GFFT = gpyfft.GpyFFT()

class FFT(object):
    def __init__(self, context, queue, input_arrays, output_arrays, axes = None, direction = 'forward', fast_math = True):
        self.context = context
        self.queue = queue

        in_array = input_arrays[0]
        out_array = output_arrays[0]

        t_strides_in, t_distance_in, t_batchsize_in, t_shape = self.calculate_transform_strides(
            axes, in_array.shape, in_array.strides, in_array.dtype)
        t_strides_out, t_distance_out, foo, bar = self.calculate_transform_strides(
            axes, out_array.shape, out_array.strides, out_array.dtype)

        plan = GFFT.create_plan(context, t_shape)
        plan.inplace = False
        plan.strides_in = t_strides_in
        plan.strides_out = t_strides_out
        plan.distances = (t_distance_in, t_distance_out)
        plan.batch_size = t_batchsize_in #assert t_batchsize_in == t_batchsize_out
        
        print 'axes', axes        
        print 'in_array.shape:          ', in_array.shape
        print 'in_array.strides/itemsize', tuple(s/in_array.dtype.itemsize for s in in_array.strides)
        print 'shape transform          ', t_shape
        print 't_strides                ', t_strides_in
        print 'distance_in              ', t_distance_in
        print 'batchsize                ', t_batchsize_in
        print

        plan.bake(self.queue)

        self.plan = plan
        self.data = in_array
        self.result = out_array
        

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
        
        axes_transform = set(a + ddim if a<0 else a for a in axes)


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

    def execute(self):

        self.plan.enqueue_transform((self.queue,), (self.data.data,), (self.result.data))
        queue.finish()

    def update_arrays(input_array, output_array):
        pass

#complex transform: 2x input_arrays real or 1x input_array, same output
#real to complex: (forward) out_array.shape[axes][-1] = in_array.shape[axes][-1]//2 + 1

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

context = cl.create_some_context()
queue = cl.CommandQueue(context)

nd_data = np.array([[1,2,3,4], [5,6,7,8]], dtype = np.complex64)
cl_data = cla.to_device(queue, nd_data)
#data = nd_data
data = cl_data
dataT = cla.to_device(queue, nd_data.T)

nd_result = np.zeros_like(nd_data, dtype = np.complex64)
cl_result = cla.to_device(queue, nd_result)
result = cl_result

axes_list = [(0,), (1,), (0,1), (1,0)]
for axes in axes_list:
    transform = FFT(context, queue, (data,), (result,), axes = axes)
    transform.execute()
    print 'data'
    print data
    print 'result'
    print result
    print 'npfftn'
    print npfftn(nd_data, axes = axes)
    print '======='
    print 

# FFT(context, (dataT,), (result,), axes = (0,))
# FFT(context, (dataT,), (result,), axes = (1,))
# FFT(context, (dataT,), (result,), axes = (0,1,))
# FFT(context, (dataT,), (result,), axes = (1,0,))


# #################
# #check for block-contiguous -> axis can be collapsed

# b = arange(24)
# b.shape = (2,3,4)

# #transform along last axis
# a[:,:,0] #indices of individual transform subarrays
# a[:,0,:]

# #nditer, nested_iters, 
