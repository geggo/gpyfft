# -*- coding: latin-1 -*-
"""
.. module:: gpyfft
   :platform: Windows, Linux
   :synopsis: A Python wrapper for the OpenCL FFT library clFFT

.. moduleauthor:: Gregor Thalhammer
"""

import cython
import pyopencl as cl
from libc.stdlib cimport malloc, free

ctypedef long int voidptr_t

DEF MAX_QUEUES = 5
DEF MAX_WAITFOR_EVENTS = 10

error_dict = {
    CLFFT_SUCCESS: 'no error',
    CLFFT_BUGCHECK: 'Bugcheck',
    CLFFT_NOTIMPLEMENTED: 'Functionality is not implemented yet.',
    CLFFT_TRANSPOSED_NOTIMPLEMENTED: 'Transposed functionality is not implemented for this transformation.',
    CLFFT_FILE_NOT_FOUND: 'Tried to open an existing file on the host system, but failed.',
    CLFFT_FILE_CREATE_FAILURE: 'Tried to create a file on the host system, but failed.',
    CLFFT_VERSION_MISMATCH: 'Version conflict between client and library.',
    CLFFT_INVALID_PLAN: 'Invalid plan.',
    CLFFT_DEVICE_NO_DOUBLE: 'Double precision not supported on this device.',
    CLFFT_DEVICE_MISMATCH: 'Attempt to run on a device using a plan baked for a different device',
    }

class GpyFFT_Error(Exception):
    """Exception wrapper for errors returned from underlying library calls"""
    def __init__(self, errorcode):
        self.errorcode = errorcode

    def __str__(self):
        error_desc = error_dict.get(self.errorcode)
        if error_desc is None:
            try:
                error_desc = cl.status_code.to_string(self.errorcode)
            except ValueError:
                error_desc = "unknown error %d", self.errorcode
        return repr(error_desc)

cdef inline bint errcheck(clfftStatus result) except True:
    cdef bint is_error = (result != CLFFT_SUCCESS)
    if is_error:
        raise GpyFFT_Error(result)
    return is_error

#main class
#TODO: need to initialize (and destroy) at module level
cdef class GpyFFT(object):
    """The GpyFFT object is the primary interface to the clFFT library"""
    def __cinit__(self, debug = False):
        cdef clfftSetupData setup_data
        errcheck(clfftInitSetupData(&setup_data))
        if debug:
            setup_data.debugFlags |= CLFFT_DUMP_PROGRAMS
        errcheck(clfftSetup(&setup_data))

    def __dealloc__(self):
        errcheck(clfftTeardown())

    def get_version(self):
        """returns the version of the underlying clFFT library

        Parameters
        ----------
            None

        Returns
        -------
            out : tuple
                the major, minor, and patch level of the clFFT library

        Raises
        ------
        GpyFFT_Error
                An error occurred accessing the clfftGetVersion function
                
        Notes
        -----
            The underlying clFFT call is 'clfftCreateDefaultPlan'
        """

        cdef cl_uint major, minor, patch
        errcheck(clfftGetVersion(&major, &minor, &patch))
        return (major, minor, patch)
    
    def create_plan(self, context, tuple shape):
        """creates an FFT Plan object based on the requested dimensionality

        Parameters
        ----------
        context : `pypencl.Context`

        shape : tuple of int
            containing from one to three integers, specifying the
            length of each requested dimension of the FFT

        Returns
        -------
        plan : `Plan`
            The generated gpyfft.Plan.

        Raises
        ------
            ValueError
                when `shape` isn't a tuple of length 1, 2 or 3
            TypeError
                when the context argument is not a `pyopencl.Context`

        """

        return Plan(context, shape, self)

#@cython.internal
cdef class Plan(object):
    """A plan is the collection of (almost) all parameters needed to specify 
    an FFT computation. This includes:

    * What pyopencl context executes the transform?
    * Is this a 1D, 2D or 3D transform?
    * What are the lengths or extents of the data in each dimension?
    * How many datasets are being transformed?
    * What is the data precision?
    * Should a scaling factor be applied to the transformed data?
    * Does the output transformed data replace the original input data in the same buffer (or buffers), or is the output data written to a different buffer (or buffers).
    * How is the input data stored in its data buffers?
    * How is the output data stored in its data buffers?

    The plan does not include:

    * The pyopencl handles to the input and output data buffers.
    * The pyopencl handle to a temporary scratch buffer (if needed).
    * Whether to execute a forward or reverse transform.

    These are specified later, when the plan is executed.
    """

    cdef clfftPlanHandle plan
    cdef object lib

    def __dealloc__(self):
        if self.plan:
            errcheck(clfftDestroyPlan(&self.plan))

    def __cinit__(self):
        self.plan = 0

    def __init__(self, context, tuple shape, lib):
        """Instantiates a Plan object

        Plan objects are created internally by gpyfft; normally
        a user does not create these objects

        Parameters
        ----------
        contex : pyopencl.Context
               http://documen.tician.de/pyopencl/runtime.html#pyopencl.Context

        shape  : tuple
               the dimensionality of the transform

        lib    :  no idea
               this is a thing that does lib things

        Raises
        ------
            ValueError
                when the shape isn't a tuple of length 1, 2 or 3
            TypeError
                because the context argument isn't a valid pyopencl.Context

        Notes
        -----
            The underlying clFFT call is 'clfftCreateDefaultPlan'
        """
    
        self.lib = lib
        if not isinstance(context, cl.Context):
            raise TypeError('expected cl.Context as type of first argument')
        
        cdef cl_context context_handle = <cl_context><voidptr_t>context.int_ptr

        ndim = len(shape)
        if ndim not in (1,2,3):
            raise ValueError('expected shape to be tuple of length 1,2 or 3')

        cdef size_t lengths[3]
        cdef int i
        for i in range(ndim):
            lengths[i] = shape[i]
        
        cdef clfftDim ndim_cl
        if ndim==1:
            ndim_cl = CLFFT_1D
        elif ndim==2:
            ndim_cl = CLFFT_2D
        elif ndim==3:
            ndim_cl = CLFFT_3D

        clfftCreateDefaultPlan(&self.plan, context_handle, ndim_cl, &lengths[0])

    property precision:
        """the floating point precision of the FFT data"""    
        def __get__(self):
            cdef clfftPrecision precision
            errcheck(clfftGetPlanPrecision(self.plan, &precision))
            return precision
        def __set__(self, clfftPrecision value):
            errcheck(clfftSetPlanPrecision(self.plan, value))

    property scale_forward:
        """the scaling factor to be applied to the FFT data for forward transforms"""    
        def __get__(self):
            cdef cl_float scale
            errcheck(clfftGetPlanScale(self.plan, CLFFT_FORWARD, &scale))
            return scale
        def __set__(self, cl_float value):
            errcheck(clfftSetPlanScale(self.plan, CLFFT_FORWARD, value))

    property scale_backward:
        """the scaling factor to be applied to the FFT data for backward transforms"""        
        def __get__(self):
            cdef cl_float scale
            errcheck(clfftGetPlanScale(self.plan, CLFFT_BACKWARD, &scale))
            return scale
        def __set__(self, cl_float value):
            errcheck(clfftSetPlanScale(self.plan, CLFFT_BACKWARD, value))

    property batch_size:
        """the number of discrete arrays that this plan can handle concurrently"""    
        def __get__(self):
            cdef size_t nbatch
            errcheck(clfftGetPlanBatchSize(self.plan, &nbatch))
            return nbatch
        def __set__(self, nbatch):
            errcheck(clfftSetPlanBatchSize(self.plan, nbatch))

    cdef clfftDim get_dim(self):
        cdef clfftDim dim
        cdef cl_uint size
        errcheck(clfftGetPlanDim(self.plan, &dim, &size))
        return dim
            
    property shape:
        """the length of each dimension of the FFT"""
        def __get__(self):
            cdef clfftDim dim = self.get_dim()
            cdef size_t sizes[3]
            errcheck(clfftGetPlanLength(self.plan, dim, &sizes[0]))
            if dim == 1:
                return (sizes[0],)
            elif dim == 2:
                return (sizes[0], sizes[1])
            elif dim == 3:
                return (sizes[0], sizes[1], sizes[2])

        def __set__(self, tuple shape):
            assert len(shape) <= 3
            cdef clfftDim dim = <clfftDim>len(shape)
            #errcheck(clfftSetPlanDim(self.plan, dim))
            cdef size_t sizes[3]
            cdef int i
            for i in range(len(shape)):
                sizes[i] = shape[i]
            errcheck(clfftSetPlanLength(self.plan, dim, &sizes[0]))

    property strides_in:
        """the distance between consecutive elements for input buffers
        in a dimension"""    
        def __get__(self):
            cdef clfftDim dim = self.get_dim()
            cdef size_t strides[3]
            errcheck(clfftGetPlanInStride(self.plan, dim, strides))
            if dim == 1:
                return (strides[0],)
            elif dim == 2:
                return (strides[0], strides[1])
            elif dim == 3:
                return (strides[0], strides[1], strides[2])

        def __set__(self, tuple strides):
            assert len(strides) <= 3
            cdef clfftDim dim = <clfftDim>len(strides)
            cdef size_t c_strides[3]
            cdef int i
            for i in range(dim):
                c_strides[i] = strides[i]
            errcheck(clfftSetPlanInStride(self.plan, dim, &c_strides[0]))

    property strides_out:
        """the distance between consecutive elements for output buffers 
        in a dimension"""        
        def __get__(self):            
            cdef clfftDim dim = self.get_dim()
            cdef size_t strides[3]
            errcheck(clfftGetPlanOutStride(self.plan, dim, strides))
            if dim == 1:
                return (strides[0],)
            elif dim == 2:
                return (strides[0], strides[1])
            elif dim == 3:
                return (strides[0], strides[1], strides[2])

        def __set__(self, tuple strides):
            assert len(strides) <= 3
            cdef clfftDim dim = <clfftDim>len(strides)
            cdef size_t c_strides[3]
            cdef int i
            for i in range(dim):
                c_strides[i] = strides[i]
            errcheck(clfftSetPlanOutStride(self.plan, dim, &c_strides[0]))
            
    property distances:
        """the distance between array objects"""    
        def __get__(self):
            cdef size_t dist_in, dist_out
            errcheck(clfftGetPlanDistance(self.plan, &dist_in, &dist_out))
            return (dist_in, dist_out)
        def __set__(self, tuple distances):
            assert len(distances) == 2
            errcheck(clfftSetPlanDistance(self.plan, distances[0], distances[1]))

    # set layout by string
    _map_layouts = {
        'COMPLEX_INTERLEAVED': CLFFT_COMPLEX_INTERLEAVED,
        'COMPLEX_PLANAR': CLFFT_COMPLEX_PLANAR,
        'HERMITIAN_INTERLEAVED': CLFFT_HERMITIAN_INTERLEAVED,
        'HERMITIAN_PLANAR': CLFFT_HERMITIAN_PLANAR,
        'REAL': CLFFT_REAL,
        }
    _enum_layouts = ( CLFFT_COMPLEX_INTERLEAVED,
                      CLFFT_COMPLEX_PLANAR,
                      CLFFT_HERMITIAN_INTERLEAVED,
                      CLFFT_HERMITIAN_PLANAR,
                      CLFFT_REAL )
    property layouts:
        """the expected layout of the output buffers"""        
        def __get__(self):
            cdef clfftLayout layout_in, layout_out
            errcheck(clfftGetLayout(self.plan, &layout_in, &layout_out))
            return (layout_in, layout_out)
        def __set__(self, tuple layouts):
            assert len(layouts) == 2
            if layouts[0] not in self._enum_layouts:
                layouts[0] = self._map_layouts[layouts[0]]
            if layouts[1] not in self._enum_layouts:
                layouts[1] = self._map_layouts[layouts[1]]
            errcheck(clfftSetLayout(self.plan, layouts[0], layouts[1]))
        
    property inplace:
        """determines if the input buffers are going to be overwritten with 
        results (True == inplace, False == out of place)"""    
        def __get__(self):
            cdef clfftResultLocation placeness
            errcheck(clfftGetResultLocation(self.plan, &placeness))
            return placeness == CLFFT_INPLACE
        def __set__(self, value):
            cdef clfftResultLocation placeness
            if value:
                placeness = CLFFT_INPLACE
            else:
                placeness = CLFFT_OUTOFPLACE
            errcheck(clfftSetResultLocation(self.plan, placeness))

    property temp_array_size:
        """Buffer size (in bytes), which may be needed internally for
        an intermediate buffer. Requires that transform plan is baked
        before."""    
        def __get__(self):
            cdef size_t buffersize
            errcheck(clfftGetTmpBufSize(self.plan, &buffersize))
            return buffersize

    property transpose_result:
        """the final transpose setting of a multi-dimensional FFT
        
        True: transpose the final result (default)
        False: skip final transpose
        """    
        def __get__(self):
            cdef clfftResultTransposed transposed
            errcheck(clfftGetPlanTransposeResult(self.plan, &transposed))
            return transposed == CLFFT_TRANSPOSED
        def __set__(self, transpose):
            cdef clfftResultTransposed transposed
            if transpose:
                transposed = CLFFT_TRANSPOSED
            else:
                transposed = CLFFT_NOTRANSPOSE
            errcheck(clfftSetPlanTransposeResult(self.plan, transposed))

    def bake(self, queues):
        """Prepare the plan for execution.

        Prepares and compiles OpenCL kernels internally used to
        perform the transform. At this point, the clfft runtime
        applies all implemented optimizations, possibly including
        running kernel experiments on the devices in the plan
        context. This can take a long time to execute. If not called,
        this is performed when the plan is execute for the first time.

        Parameters
        ----------
            queues : `pyopencl.CommandQueue` or list of `pyopencl.CommandQueue`

        Returns
        -------
            None

        Raises
        ------
            `GpyFFT_Error`
                An error occurred accessing the clfftBakePlan function

        Notes
        -----
            The underlying clFFT call is 'clfftBakePlan'
        """

        if isinstance(queues, cl.CommandQueue):
            queues = (queues,)
        cdef int n_queues = len(queues)
        assert n_queues <= MAX_QUEUES
        cdef cl_command_queue queues_[MAX_QUEUES]
        cdef int i
        for i in range(n_queues):
            assert isinstance(queues[i], cl.CommandQueue)
            queues_[i] = <cl_command_queue><voidptr_t>queues[i].int_ptr
        errcheck(clfftBakePlan(self.plan,
                                  n_queues, queues_,
                                  NULL, NULL))

    def set_callback(self,
                     func_name,
                     func_string,
                     callback_type,
                     local_mem_size=0,
                     user_data=None):
        """Register callback.

        Parameters
        ----------
        func_name: bytes
            callback function name

        func_string: bytes
            callback function, gets inlined in OpenCL kernel

        callback_type: 'pre' or 'post'

        local_mem_size: int
            size (bytes) of the local memory used by the callback

        user_data:
            pyopencl.Buffer or iterable of pyopencl.Buffer

        Notes
        -----
            The underlying clFFT call 'clSetPlanCallback'

        """

        typedict = {'pre': PRECALLBACK,
                    'post': POSTCALLBACK}
        clfft_callback_type = typedict[callback_type]

        if user_data is None:
            user_data = ()
            
        if isinstance(user_data, cl.Buffer):
            user_data = (user_data,)

        n_user_data_buffers = len(user_data)
        
        cdef cl_mem* user_buffers = NULL
        if n_user_data_buffers:
            user_buffers = <cl_mem*>malloc(n_user_data_buffers*sizeof(cl_mem))
            for n, user_data_buffer in enumerate(user_data):
                assert isinstance(user_data_buffer, cl.Buffer)
                user_buffers[n] = <cl_mem><voidptr_t>user_data_buffer.int_ptr

        try:
            res = clfftSetPlanCallback(self.plan,
                                       func_name,
                                       func_string,
                                       local_mem_size,
                                       clfft_callback_type,
                                       user_buffers,
                                       n_user_data_buffers)
        finally:
            free(user_buffers)
        errcheck(res)

    def enqueue_transform(self, 
                         queues, 
                         in_buffers, 
                         out_buffers = None,
                         direction_forward = True, 
                         wait_for_events = None, 
                         temp_buffer = None,
                         ):
        """Enqueue an FFT transform operation, and return immediately.

        Parameters
        ----------
        queues : pyopencl.CommandQueue or iterable of pyopencl.CommandQueue

        in_buffers : pyopencl.Buffer or iterable (1 or 2 items) of pyopencl.Buffer

        out_buffers : pyopencl.Buffer or iterable (1 or 2 items) of pyopencl.Buffer, optional
            can be None for inplace transforms

        Other Parameters
        ----------------

        direction_forward : bool, optional
            Perform forward transform (default True).

        wait_for_events : iterable of pyopencl.Event, optional
            Ensures that all events in this list have finished
            execution before transform is performed.

        temp_buffer : pyopencl.Buffer, optional
            For intermediate results a temporary buffer can be
            provided. The size (in bytes) of this buffer is given by
            the `temp_array_size` property.

        Returns
        -------
            tuple of `pyopencl.Event`, one event for each command queue in `queues`

        Raises
        ------
            `GpyFFT_Error`
                An error occurred accessing the clfftEnqueueTransform function

        Notes
        -----
            The underlying clFFT call is 'clfftEnqueueTransform'
        """

        cdef int i

        cdef clfftDirection direction
        if direction_forward:
            direction = CLFFT_FORWARD
        else:
            direction = CLFFT_BACKWARD
            
        cdef cl_command_queue queues_[MAX_QUEUES]
        if isinstance(queues, cl.CommandQueue):
            queues = (queues,)
        n_queues = len(queues)
        assert n_queues <= MAX_QUEUES
        for i, queue in enumerate(queues):
            assert isinstance(queue, cl.CommandQueue)
            queues_[i] = <cl_command_queue><voidptr_t>queue.int_ptr
            
        cdef cl_event wait_for_events_array[MAX_WAITFOR_EVENTS]
        cdef cl_event* wait_for_events_ = NULL
        cdef n_waitfor_events = 0
        if wait_for_events is not None:
            n_waitfor_events = len(wait_for_events)
            assert n_waitfor_events <= MAX_WAITFOR_EVENTS
            for i, event in enumerate(wait_for_events):
                assert isinstance(event, cl.Event)
                wait_for_events_array[i] = <cl_event><voidptr_t>event.int_ptr
            wait_for_events_ = &wait_for_events_array[0]

        cdef cl_mem in_buffers_[2]
        if isinstance(in_buffers, cl.Buffer):
            in_buffers = (in_buffers,)
        n_in_buffers = len(in_buffers)
        assert n_in_buffers <= 2
        for i, in_buffer in enumerate(in_buffers):
            assert isinstance(in_buffer, cl.Buffer)
            in_buffers_[i] = <cl_mem><voidptr_t>in_buffer.int_ptr

        cdef cl_mem out_buffers_array[2]
        cdef cl_mem* out_buffers_ = NULL
        if out_buffers is not None:
            if isinstance(out_buffers, cl.Buffer):
                out_buffers = (out_buffers,)
            n_out_buffers = len(out_buffers)
            assert n_out_buffers in (1,2)
            for i, out_buffer in enumerate(out_buffers):
                assert isinstance(out_buffer, cl.Buffer)
                out_buffers_array[i] = <cl_mem><voidptr_t>out_buffer.int_ptr
            out_buffers_ = &out_buffers_array[0]

        cdef cl_mem tmp_buffer_ = NULL
        if temp_buffer is not None:
            assert isinstance(temp_buffer, cl.Buffer)
            tmp_buffer_ = <cl_mem><voidptr_t>temp_buffer.int_ptr

        cdef cl_event out_cl_events[MAX_QUEUES]

        errcheck(clfftEnqueueTransform(self.plan,
                                          direction,
                                          n_queues,
                                          &queues_[0],
                                          n_waitfor_events,
                                          &wait_for_events_[0],
                                          out_cl_events,
                                          &in_buffers_[0],
                                          out_buffers_,
                                          tmp_buffer_))
        
        #return tuple((cl.Event.from_cl_event_as_int(<long>out_cl_events[i]) for i in range(n_queues)))
        return tuple((cl.Event.from_int_ptr(<long>out_cl_events[i], retain=False) for i in range(n_queues)))
            
        
            
        
        

#gpyfft = GpyFFT()


#cdef Plan PlanFactory():
    #cdef Plan instance = Plan.__new__(Ref)
    #instance.plan = None
    #return instance
