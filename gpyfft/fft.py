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

        # if no axes are given, transform all axes, select axes order for good performance depending on memory layout
        if axes is None:
            if in_array.flags.c_contiguous:
                axes = np.arange(in_array.ndim)[::-1]
            elif in_array.flags.f_contiguous:
                axes = np.arange(in_array.ndim)
            else:
                axes = np.arange(in_array.ndim)[::-1]
                # TODO: find good heuristics for this (rare), e.g. based on strides
        else:
            axes = np.asarray(axes)
            
        t_strides_in, t_distance_in, t_batchsize_in, t_shape, axes_transform = self.calculate_transform_strides(axes, in_array)

        if out_array is not None:
            t_inplace = False
            t_strides_out, t_distance_out, t_batchsize_out, t_shape_out, axes_transform_out = self.calculate_transform_strides(
                axes, out_array)
            if in_array.base_data is out_array.base_data:
                t_inplace = True

            #assert t_batchsize_out == t_batchsize_in and t_shape == t_shape_out, 'input and output size does not match' #TODO: fails for real-to-complex
            assert np.all(axes_transform == axes_transform_out), 'error finding transform axis (consider setting axes argument)'
            
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

        if t_inplace and ((layout_in is gfft.CLFFT_REAL) or
                          (layout_out is gfft.CLFFT_REAL)):
            assert ((in_array.strides[axes_transform[0]] == in_array.dtype.itemsize) and \
                    (out_array.strides[axes_transform[0]] == out_array.dtype.itemsize)), \
                    'inline real transforms need stride 1 for first transform axis'


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
            print('layout_in                ', str(layout_in).split('.')[1])
            print('t_strides                ', t_strides_in)
            print('distance_in              ', t_distance_in)
            print('batchsize                ', t_batchsize_in)
            print('layout_out               ', str(layout_out).split('.')[1])
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
    def calculate_transform_strides(cls, axes_transform, array):
    
        shape = np.array(array.shape)
        strides = np.array(array.strides)
        dtype = array.dtype
        
        ddim = len(shape) #dimensionality data        
        tdim = len(axes_transform) #dimensionality transform
        assert tdim <= ddim

        # transform negative axis values (e.g. -1 for last axis) to positive
        axes_transform[axes_transform<0] += ddim
        
        # remaining, non-transformed axes
        axes_notransform = np.setdiff1d(range(ddim), axes_transform)
        
        # sort non-transformed axes by strides. [::-1] takes care of unit-size dimensions in the middle of the shape.
        axes_notransform = axes_notransform[np.argsort(strides[axes_notransform][::-1])][::-1]
        
        #print "axes_notransformed sorted", axes_notransform
        
        # -> list of collapsable axes, [ [x,y], [z] ]
        collapsable_axes_list = [] #result
        collapsable_axes_candidates = axes_notransform[:1].tolist() #intermediate list of collapsable axes (magic code to get empty list if axes_notransform is empty)
        for a in axes_notransform[1:]:
            if strides[a] == strides[collapsable_axes_candidates[-1]] * shape[collapsable_axes_candidates[-1]]:
                collapsable_axes_candidates.append(a) #add axes to intermediate list of collapsable axes
            else: #does not fit into current intermediate list of collapsable axes
                collapsable_axes_list.append(collapsable_axes_candidates) #store away intermediate list
                collapsable_axes_candidates = [a] #start new intermediate list
        collapsable_axes_list.append(collapsable_axes_candidates) #append last intermediate list to 
        
        assert len(collapsable_axes_list) == 1, 'data layout not supported (only single non-transformed axis allowd)' #all non-transformed axes collapsed
        axes_notransform = collapsable_axes_list[0] #all axes collapsable: take single group of collapsable axes
        
        t_distances = strides[axes_notransform]//dtype.itemsize
                
        if len(t_distances) == 0:
            t_distance = 0
        else:
            t_distance = t_distances[0] #takes smalles stride (axes_notransform have been sorted by stride size)
                       
        batchsize = np.prod(shape[axes_notransform])
        
        t_shape = shape[axes_transform]
        t_strides = strides[axes_transform]//dtype.itemsize
        
        return (tuple(t_strides), t_distance, batchsize, tuple(t_shape), tuple(axes_transform)) #, tuple(axes_notransform))


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
