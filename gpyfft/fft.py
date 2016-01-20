from __future__ import absolute_import, division, print_function
from .gpyfftlib import GpyFFT
import pyopencl as cl
GFFT = GpyFFT(debug=False)

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
            print('axes', axes        )
            print('in_array.shape:          ', in_array.shape)
            print('in_array.strides/itemsize', tuple(s // in_array.dtype.itemsize for s in in_array.strides))
            print('shape transform          ', t_shape)
            print('t_strides                ', t_strides_in)
            print('distance_in              ', t_distance_in)
            print('batchsize                ', t_batchsize_in)
            print('t_stride_out             ', t_strides_out)
            print('inplace                  ', t_inplace)

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

    def update_arrays(self, input_array, output_array):
        pass
