class FFT(object):
    def __init__(self, input_arrays, output_arrays = None, axes = None, direction = 'forward', fast_math = True):
        
        in_array = input_arrays[0]

        #assert len(axes_notransform) < 2, 'more than one non-transformed axis'
        #TODO: collapse non-transformed axes if possible
        #out_array = output_arrays[0]
        self.calculate_transform_strides(in_array.shape, in_array.dtype, in_array.strides, axes)
                
    def calculate_transform_strides(self, shape_in, dtype_in, strides_in, axes):
        ddim = len(shape_in) #dimensionality data
        if axes is None:
            axes = range(ddim)
        tdim = len(axes) #dimensionality transform
        assert tdim <= ddim
        
        axes_transform = set(a + ddim if a<0 else a for a in axes)
        axes_notransform = set(range(ddim)).difference(axes_transform)

        dsize = dtype_in.itemsize
        strides_transform = [strides_in[i]//dsize for i in axes_transform]
        distance_transform = [strides_in[i]//dsize for i in axes_notransform]
        
        batchsize = 1
        for a in axes_notransform:
            batchsize *= shape_in[a]

        print 'in_array.shape:', shape_in
        print 'axes', axes
        print 'in_array.strides/itemsize', tuple(s/dsize for s in strides_in)
        print 'axes transform:', axes_transform
        print 'axes no transform:', axes_notransform

        print 'strides_in', strides_transform
        print 'distance_in', distance_transform
        print 'batchsize', batchsize
        print

    def execute(self):
        pass

    def update_arrays(input_array, output_array):
        pass

#complex transform: 2x input_arrays real or 1x input_array, same output
#real to complex: (forward) out_array.shape[axes][-1] = in_array.shape[axes][-1]//2 + 1

import numpy as np
import pyopencl as cl
import pyopencl.array as cla

#context = cl.create_some_context()
#queue = cl.CommandQueue(context)

nd_data = np.array([[1,2,3,4], [5,6,7,8]], dtype = np.complex64)
#cl_data = cla.to_device(queue, nd_data)
data = nd_data

FFT((data,))

FFT((data,), axes = (0,1))
FFT((data,), axes = (0,1))

FFT((data.T,), axes = (0,))
FFT((data.T,), axes = (1,))
FFT((data.T,), axes = (0,1,))
FFT((data.T,), axes = (1,0,))


#################
#check for block-contiguous -> axis can be collapsed

b = arange(24)
b.shape = (2,3,4)

#transform along last axis
a[:,:,0] #indices of individual transform subarrays
a[:,0,:]

#nditer, nested_iters, 
