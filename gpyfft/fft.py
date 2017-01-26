from __future__ import absolute_import, division, print_function
from .gpyfftlib import GpyFFT
import gpyfft.gpyfftlib as gfft
import pyopencl as cl
GFFT = GpyFFT(debug=False)

import pyopencl as cl
import numpy as np

# TODO:

class FFT(object):
    def __init__(self, context, queue, in_array, out_array=None, axes = None,
                 fast_math = False,
                 real=False,
                 callbacks=None, #dict: 'pre', 'post'
    ):
        # Callbacks: dict(pre=b'pre source (kernel named pre!)')
        self.context = context
        self.queue = queue

        t_strides_in, t_distance_in, t_batchsize_in, t_shape, axes_transform = self.calculate_transform_strides(
            axes, in_array.shape, in_array.strides, in_array.dtype)

        if out_array is not None:
            t_inplace = False
            t_strides_out, t_distance_out, t_batchsize_out, t_shape_out, foo = self.calculate_transform_strides(
                axes, out_array.shape, out_array.strides, out_array.dtype)

            #assert t_batchsize_out == t_batchsize_in and t_shape == t_shape_out, 'input and output size does not match' #TODO: fails for real-to-complex
            
        else:
            t_inplace = True
            t_strides_out, t_distance_out = t_strides_in, t_distance_in

        
        #assert np.issubclass(in_array.dtype, np.complexfloating) and \
        #    np.issubclass(in_array.dtype, np.complexfloating), \
                
        #precision (+ fast_math!)
        #complex64 <-> complex64
        #complex128 <-> complex128

        if in_array.dtype in (np.float32, np.complex64):
            precision = gfft.CLFFT_SINGLE
        elif in_array.dtype in (np.float64, np.complex128):
            precision = gfft.CLFFT_DOUBLE

        #TODO: add assertions that precision match
        if in_array.dtype in (np.float32, np.float64):
            layout_in = gfft.CLFFT_REAL
            layout_out = gfft.CLFFT_HERMITIAN_INTERLEAVED

            expected_out_shape = list(in_array.shape)
            expected_out_shape[axes_transform[0]] = expected_out_shape[axes_transform[0]]//2 + 1
            assert out_array.shape == tuple(expected_out_shape), \
                'output array shape %s does not match expected shape: %s'%(out_array.shape,expected_out_shape)

        elif in_array.dtype in (np.complex64, np.complex128):
            if not real:
                layout_in = gfft.CLFFT_COMPLEX_INTERLEAVED
                layout_out = gfft.CLFFT_COMPLEX_INTERLEAVED
            else:
                # complex-to-real transform
                layout_in = gfft.CLFFT_HERMITIAN_INTERLEAVED
                layout_out = gfft.CLFFT_REAL
                t_shape = t_shape_out

        self.t_shape = t_shape
        self.batchsize = t_batchsize_in

        plan = GFFT.create_plan(context, t_shape)
        plan.inplace = t_inplace
        plan.strides_in = t_strides_in
        plan.strides_out = t_strides_out
        plan.distances = (t_distance_in, t_distance_out)
        plan.batch_size = self.batchsize
        plan.precision = precision
        plan.layouts = (layout_in, layout_out)

        if callbacks is not None:
            if callbacks.has_key('pre'):
                plan.set_callback(b'pre',
                                  callbacks['pre'],
                                  'pre')
            if 'post' in callbacks:
                plan.set_callback(b'post',
                                  callbacks['post'],
                                  'post')
        
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
        self.data = in_array
        self.result = out_array

    @classmethod
    def calculate_transform_strides(cls,
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

        # transform negative axis values (e.g. -1 for last axis) to positive
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

        return (tuple(t_strides), t_distance, batchsize, tuple(t_shape), axes_transform)

    def enqueue(self, forward = True, wait_for_events = None):
        return self.enqueue_arrays(forward=forward, data=self.data, result=self.result, wait_for_events=wait_for_events)

    def enqueue_arrays(self, data = None, result = None, forward = True, wait_for_events = None):
        """enqueue transform"""
        if data is None:
            data = self.data
        else:
            assert data.shape == self.data.shape
            assert data.strides == self.data.strides
            assert data.dtype == self.data.dtype
        if result is None:
            result = self.result
        else:
            assert result.shape == self.result.shape
            assert result.strides == self.result.strides
            assert result.dtype == self.result.dtype

        # get buffer for data
        if data.offset != 0:
            data = data._new_with_changes(data=data.base_data[data.offset:], offset=0)
        data_buffer = data.base_data

        if result is not None:
            # get buffer for result
            if result.offset != 0:
                result = result._new_with_changes(data=result.base_data[result.offset:], offset=0)
            result_buffer = result.base_data

            events = self.plan.enqueue_transform((self.queue,), (data_buffer,), (result_buffer),
                                        direction_forward = forward, temp_buffer = self.temp_buffer, wait_for_events = wait_for_events)
        else:
            events = self.plan.enqueue_transform((self.queue,), (data_buffer,),
                                        direction_forward = forward, temp_buffer = self.temp_buffer, wait_for_events = wait_for_events)

        return events

    def update_arrays(self, input_array, output_array):
        pass
