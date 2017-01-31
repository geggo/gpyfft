from __future__ import absolute_import, division, print_function
from .gpyfftlib import GpyFFT
import gpyfft.gpyfftlib as gfft
import pyopencl as cl
GFFT = GpyFFT(debug=False)

import pyopencl as cl
import numpy as np

import sys

# TODO:

class FFT(object):
    def __init__(self, context, queue, in_array, out_array=None, axes = None,
                 fast_math = False,
                 real=False,
                 fft_shape = None, # required only for in-place complex-to-real
                 callbacks=None, #dict: 'pre', 'post'
    ):
        # Callbacks: dict(pre=b'pre source (kernel named pre!)')
        self.context = context
        self.queue = queue

        t_inplace = out_array is None

        # in case of real transforms double-check parameters and also
        # calculate expected transform sizes
        in_type = in_array.dtype
        in_kind = in_type.kind
        in_itemsize = in_type.itemsize
        assert in_kind in ('c', 'f') and in_itemsize in (4, 8, 16), "Input array must be float32, float64, complex64, or complex128"
        out_kind = None
        if not t_inplace:
            out_type = out_array.dtype
            out_kind = out_type.kind
            out_itemsize = out_type.itemsize
            assert out_kind in ('c', 'f') and out_itemsize in (4, 8, 16), "Output array must be float32, float64, complex64, or complex128"
        assert not (in_kind == 'f' and out_kind == 'f'), "Input and output can't both be real!"
        assert not (real and (in_kind == 'c' and  out_kind == 'c')), "Input and output are both complex but real=True is specified!"
        if (in_kind == 'f' or out_kind == 'f') and not real:
            real = True
            sys.stderr.write("Setting 'real' to True because input or output is real (specify real=True to avoid this warning)\n")
        if not t_inplace and not real:
            assert in_itemsize == out_itemsize, "Input and output arrays must have same precision in complex->complex transforms!"
        # all in/out datatypes are consistent with each other

        t_strides_in, t_distance_in, t_batchsize_in, t_shape, axes_transform = self.calculate_transform_strides(axes, in_array)

        if real:
            real_axis = axes_transform[0]

        if out_array is not None:
            t_strides_out, t_distance_out, t_batchsize_out, t_shape_out, foo = self.calculate_transform_strides(
                axes, out_array)

            #assert t_batchsize_out == t_batchsize_in and t_shape == t_shape_out, 'input and output size does not match' #TODO: fails for real-to-complex

        else:
            if real:
                t_strides_out = t_strides_in
                if in_kind == 'f':
                    t_distance_out = t_distance_in // 2 # real-to-complex
                else:
                    t_distance_out = t_distance_in * 2 # complex-to-real
            else:
                t_strides_out, t_distance_out = t_strides_in, t_distance_in
            
        # double-check input/output shapes for real transforms
        if real:
            if in_kind == 'f':
                # real-to-complex
                expected_out_shape = list(in_array.shape)
                expected_out_shape[real_axis] = (expected_out_shape[real_axis] // 2) + 1
                if t_inplace:
                    # In-place -- make sure there is enough padding in
                    # first transformed axis
                    # (need space for 2 * ((N // 2) + 1)) elements)
                    # We check padding by looking at the strides, since
                    # user will have sent a "sliced" version of a padded
                    # array.
                    assert real_axis + 1 < in_array.ndim, "in-place real-to-complex transforms require an extra dimension after the 'real' transformed axis (to determine if you've sent proper padding), even if that dimension is of size 1"
                    curstride = in_array.strides[real_axis + 1]
                    curstride_elems = curstride / in_itemsize
                    expectedstride_elems = (2 * expected_out_shape[real_axis])
                    expectedstride = expectedstride_elems * in_itemsize
                    assert expectedstride <= curstride, "Not enough padding in array for real-to-complex in axis %d (found next-dim stride of %d bytes (%d elements), expected %d bytes (%d elements)" % (real_axis, curstride, curstride_elems, expectedstride, expectedstride_elems)
                else:
                    assert out_array.shape == tuple(expected_out_shape), \
                        'output array shape %s does not match expected shape: %s'%(out_array.shape,expected_out_shape)
                    
                if fft_shape is not None:
                    assert tuple(fft_shape) == in_array.shape, "fft_shape set incorrectly (in any case, not necessary to send fft_shape for real-to-complex)"
            else:
                # complex-to-real
                if t_inplace:
                    # In-place -- make sure fft_shape matches input
                    assert fft_shape is not None, "fft_shape is required for in-place complex-to-real transforms!"
                    expected_in_shape = list(fft_shape)
                    expected_in_shape[real_axis] = (expected_in_shape[real_axis] // 2) + 1
                    assert in_array.shape == tuple(expected_in_shape), \
                        'input array shape %s does not match expected shape: %s'%(in_array.shape, expected_in_shape)
                else:
                    expected_in_shape = list(out_array.shape)
                    expected_in_shape[real_axis] = (expected_in_shape[real_axis] // 2) + 1
                    assert in_array.shape == tuple(expected_in_shape), \
                        'input array shape %s does not match expected shape: %s'%(in_array.shape, expected_in_shape)
                    if fft_shape is not None:
                        assert tuple(fft_shape) == expected_in_shape, "fft_shape set incorrectly (in any case, not necessary to send fft_shape for out-of-place complex-to-real)"


        #assert np.issubclass(in_array.dtype, np.complexfloating) and \
        #    np.issubclass(in_array.dtype, np.complexfloating), \
                
        #precision (+ fast_math!)
        #complex64 <-> complex64
        #complex128 <-> complex128

        if in_array.dtype in (np.float32, np.complex64):
            precision = gfft.CLFFT_SINGLE
        elif in_array.dtype in (np.float64, np.complex128):
            precision = gfft.CLFFT_DOUBLE

        if in_array.dtype in (np.float32, np.float64):
            layout_in = gfft.CLFFT_REAL
            layout_out = gfft.CLFFT_HERMITIAN_INTERLEAVED
        elif in_array.dtype in (np.complex64, np.complex128):
            if not real:
                layout_in = gfft.CLFFT_COMPLEX_INTERLEAVED
                layout_out = gfft.CLFFT_COMPLEX_INTERLEAVED
            else:
                # complex-to-real transform
                layout_in = gfft.CLFFT_HERMITIAN_INTERLEAVED
                layout_out = gfft.CLFFT_REAL
                if t_inplace:
                    t_shape = tuple((fft_shape[x] for x in axes_transform))
                else:
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
            print('fft_shape:               ', fft_shape)
            if out_array is not None:
                print('out_array.shape:          ', out_array.shape)
                print('out_array.strides/itemsize', tuple(s // out_array.dtype.itemsize for s in out_array.strides))
            print('shape transform          ', t_shape)
            print('layout_in                ', str(layout_in).split('.')[1])
            print('t_strides                ', t_strides_in)
            print('distance_in              ', t_distance_in)
            print('distance_out             ', t_distance_out)
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
    def calculate_transform_strides(cls, axes, array):
    
        shape = np.array(array.shape)
        strides = np.array(array.strides)
        dtype = array.dtype
        
        ddim = len(shape) #dimensionality data
        
        #transform along all axes if transform axes are not given (None)
        axes_transform = np.arange(ddim) if axes is None else np.array(axes)

        tdim = len(axes_transform) #dimensionality transform
        assert tdim <= ddim

        # transform negative axis values (e.g. -1 for last axis) to positive
        axes_transform[axes_transform<0] += ddim
        
        # remaining, non-transformed axes
        axes_notransform = np.lib.arraysetops.setdiff1d(range(ddim), axes_transform)
        
        #sort non-transformed axes by strides
        axes_notransform = axes_notransform[np.argsort(strides[axes_notransform])]
        
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
        
        assert len(collapsable_axes_list) == 1 #all non-transformed axes collapsed
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
