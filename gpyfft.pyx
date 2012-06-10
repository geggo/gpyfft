# -*- coding: latin-1 -*-
"""
.. module:: gpyfft
   :platform: Windows
   :synopsis: A Python wrapper for the OpenCL FFT library APPML/clAmdFft from AMD

.. moduleauthor:: Gregor Thalhammer
"""

import cython
import pyopencl as cl

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
    CLFFT_INVALID_PLAN: 'Requested plan could not be found.',
    CLFFT_DEVICE_NO_DOUBLE: 'Double precision not supported on this device.',
    }

class GpyFFT_Error(Exception):
    """Exception wrapper for errors returned from underlying AMD library calls"""
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

cdef inline bint errcheck(clAmdFftStatus result) except True:
    cdef bint is_error = (result != CLFFT_SUCCESS)
    if is_error:
        raise GpyFFT_Error(result)
    return is_error

#main class
#TODO: need to initialize (and destroy) at module level
cdef class GpyFFT(object):
    """The GpyFFT object is the primary interface to the AMD FFT library"""
    def __cinit__(self): #TODO: add debug flag
        cdef clAmdFftSetupData setup_data
        errcheck(clAmdFftInitSetupData(&setup_data))
        errcheck(clAmdFftSetup(&setup_data))

    def __dealloc__(self):
        errcheck(clAmdFftTeardown())

    def get_version(self):
        """returns the version of the underlying AMD FFT library

        Parameters
        ----------
            None

        Returns
        -------
            out : tuple
                the major, minor, and patch level of the AMD FFT library

        Raises
        ------
        GpyFFT_Error
                An error occurred accessing the clAmdFftGetVersion function
                
        Notes
        -----
            The underlying AMD FFT call is 'clAmdFftCreateDefaultPlan'
        """

        cdef cl_uint major, minor, patch
        errcheck(clAmdFftGetVersion(&major, &minor, &patch))
        return (major, minor, patch)
    
    def create_plan(self, context, tuple shape):
        r"""creates an FFT Plan object based on the requested dimensionality

        Parameters
        ----------
        context : pypencl.Context
                 http://documen.tician.de/pyopencl/runtime.html#pyopencl.Context

        shape : tuple
            containing from one to three integers, specifying the
            length of each requested dimension of the FFT

        Returns
        -------
        out : gpyfft.Plan object
            The generated gpyfft.Plan

        Raises
        ------
        None
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

    cdef clAmdFftPlanHandle plan
    cdef object lib

    def __dealloc__(self):
        if self.plan:
            errcheck(clAmdFftDestroyPlan(&self.plan))

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
            The underlying AMD FFT call is 'clAmdFftCreateDefaultPlan'
        """
    
        self.lib = lib
        if not isinstance(context, cl.Context):
            raise TypeError('expected cl.Context as type of first argument')
        
        cdef cl_context context_handle = <cl_context><voidptr_t>context.obj_ptr

        ndim = len(shape)
        if ndim not in (1,2,3):
            raise ValueError('expected shape to be tuple of length 1,2 or 3')

        cdef size_t lengths[3]
        cdef int i
        for i in range(ndim):
            lengths[i] = shape[i]
        
        clAmdFftCreateDefaultPlan(&self.plan, context_handle, ndim, &lengths[0])

    property precision:
        """the floating point precision of the FFT data"""    
        def __get__(self):
            cdef clAmdFftPrecision precision
            errcheck(clAmdFftGetPlanPrecision(self.plan, &precision))
            return precision
        def __set__(self, clAmdFftPrecision value):
            errcheck(clAmdFftSetPlanPrecision(self.plan, value))

    property scale_forward:
        """the scaling factor to be applied to the FFT data for forward transforms"""    
        def __get__(self):
            cdef cl_float scale
            errcheck(clAmdFftGetPlanScale(self.plan, CLFFT_FORWARD, &scale))
            return scale
        def __set__(self, cl_float value):
            errcheck(clAmdFftSetPlanScale(self.plan, CLFFT_FORWARD, value))

    property scale_backward:
        """the scaling factor to be applied to the FFT data for backward transforms"""        
        def __get__(self):
            cdef cl_float scale
            errcheck(clAmdFftGetPlanScale(self.plan, CLFFT_BACKWARD, &scale))
            return scale
        def __set__(self, cl_float value):
            errcheck(clAmdFftSetPlanScale(self.plan, CLFFT_BACKWARD, value))

    property batch_size:
        """the number of discrete arrays that this plan can handle concurrently"""    
        def __get__(self):
            cdef size_t nbatch
            errcheck(clAmdFftGetPlanBatchSize(self.plan, &nbatch))
            return nbatch
        def __set__(self, nbatch):
            errcheck(clAmdFftSetPlanBatchSize(self.plan, nbatch))

    cdef clAmdFftDim get_dim(self):
        """retrieve the dimensionality of FFTs to be transformed in the plan

        Parameters
        ----------
            None

        Returns
        -------
            out : tuple
                the major, minor, and patch level of the AMD FFT library

        Raises
        ------
        GpyFFT_Error
                An error occurred accessing the clAmdFftGetPlanDim function
                
        Notes
        -----
            The underlying AMD FFT call is 'clAmdFftGetPlanDim'
        """

        cdef clAmdFftDim dim
        cdef cl_uint size
        errcheck(clAmdFftGetPlanDim(self.plan, &dim, &size))
        return dim
            
    property shape:
        """the length of each dimension of the FFT"""    
        def __get__(self):
            cdef clAmdFftDim dim = self.get_dim()
            cdef size_t sizes[3]
            errcheck(clAmdFftGetPlanLength(self.plan, dim, &sizes[0]))
            if dim == 1:
                return (sizes[0],)
            elif dim == 2:
                return (sizes[0], sizes[1])
            elif dim == 3:
                return (sizes[0], sizes[1], sizes[2])

        def __set__(self, tuple shape):
            assert len(shape) <= 3
            cdef clAmdFftDim dim = <clAmdFftDim>len(shape)
            #errcheck(clAmdFftSetPlanDim(self.plan, dim))
            cdef size_t sizes[3]
            cdef int i
            for i in range(len(shape)):
                sizes[i] = shape[i]
            errcheck(clAmdFftSetPlanLength(self.plan, dim, &sizes[0]))

    property strides_in:
        """the distance between consecutive elements for input buffers 
        in a dimension"""    
        def __get__(self):
            cdef clAmdFftDim dim = self.get_dim()
            cdef size_t strides[3]
            errcheck(clAmdFftGetPlanInStride(self.plan, dim, strides))
            if dim == 1:
                return (strides[0],)
            elif dim == 2:
                return (strides[0], strides[1])
            elif dim == 3:
                return (strides[0], strides[1], strides[2])

        def __set__(self, tuple strides):
            assert len(strides) <= 3
            cdef clAmdFftDim dim = <clAmdFftDim>len(strides)
            cdef size_t c_strides[3]
            cdef int i
            for i in range(dim):
                c_strides[i] = strides[i]
            errcheck(clAmdFftSetPlanInStride(self.plan, dim, &c_strides[0]))

    property strides_out:
        """the distance between consecutive elements for output buffers 
        in a dimension"""        
        def __get__(self):            
            cdef clAmdFftDim dim = self.get_dim()
            cdef size_t strides[3]
            errcheck(clAmdFftGetPlanOutStride(self.plan, dim, strides))
            if dim == 1:
                return (strides[0],)
            elif dim == 2:
                return (strides[0], strides[1])
            elif dim == 3:
                return (strides[0], strides[1], strides[2])

        def __set__(self, tuple strides):
            assert len(strides) <= 3
            cdef clAmdFftDim dim = <clAmdFftDim>len(strides)
            cdef size_t c_strides[3]
            cdef int i
            for i in range(dim):
                c_strides[i] = strides[i]
            errcheck(clAmdFftSetPlanOutStride(self.plan, dim, &c_strides[0]))
            
    property distances:
        """the distance between array objects"""    
        def __get__(self):
            cdef size_t dist_in, dist_out
            errcheck(clAmdFftGetPlanDistance(self.plan, &dist_in, &dist_out))
            return (dist_in, dist_out)
        def __set__(self, tuple distances):
            assert len(distances) == 2
            errcheck(clAmdFftSetPlanDistance(self.plan, distances[0], distances[1]))

    property layouts:
        """the expected layout of the output buffers"""        
        def __get__(self):
            cdef clAmdFftLayout layout_in, layout_out
            errcheck(clAmdFftGetLayout(self.plan, &layout_in, &layout_out))
            return (layout_in, layout_out)
        def __set__(self, tuple layouts):
            assert len(layouts) == 2
            errcheck(clAmdFftSetLayout(self.plan, layouts[0], layouts[1]))
        
    property inplace:
        """determines if the input buffers are going to be overwritten with 
        results (True == inplace, False == out of place)"""    
        def __get__(self):
            cdef clAmdFftResultLocation placeness
            errcheck(clAmdFftGetResultLocation(self.plan, &placeness))
            return placeness == CLFFT_INPLACE
        def __set__(self, value):
            cdef clAmdFftResultLocation placeness
            if value:
                placeness = CLFFT_INPLACE
            else:
                placeness = CLFFT_OUTOFPLACE
            errcheck(clAmdFftSetResultLocation(self.plan, placeness))

    property temp_array_size:
        """the buffer size (in bytes), which may be needed internally for an
        intermediate buffer"""    
        def __get__(self):
            cdef size_t buffersize
            errcheck(clAmdFftGetTmpBufSize(self.plan, &buffersize))
            return buffersize

    property transpose_result:
        """the final transpose setting of a multi-dimensional FFT"""    
        def __get__(self):
            cdef clAmdFftResultTransposed transposed
            errcheck(clAmdFftGetPlanTransposeResult(self.plan, &transposed))
            return transposed == CLFFT_TRANSPOSED
        def __set__(self, transpose):
            cdef clAmdFftResultTransposed transposed
            if transpose:
                transposed = CLFFT_TRANSPOSED
            else:
                transposed = CLFFT_NOTRANSPOSE
            errcheck(clAmdFftSetPlanTransposeResult(self.plan, transposed))

    def bake(self, queues):
        """Prepare the plan for execution

        After all plan parameters are set, the client has the option of "baking" the plan, which tells the
        runtime no more changes to the plan's parameters are expected, and the OpenCL kernels are
        to be compiled. This optional function allows the client application to perform this function when
        the application is being initialized instead of on the first execution. At this point, the clAmdFft
        runtime applies all implemented optimizations, possibly including running kernel experiments on
        the devices in the plan context.

        Parameters
        ----------
            queues : list
                   this is a list of things

        Returns
        -------
            None

        Raises
        ------
            GpyFFT_Error
                An error occurred accessing the clAmdFftBakePlan function

        Notes
        -----
            The underlying AMD FFT call is 'clAmdFftBakePlan'
        """

        if isinstance(queues, cl.CommandQueue):
            queues = (queues,)
        cdef int n_queues = len(queues)
        assert n_queues <= MAX_QUEUES
        cdef cl_command_queue queues_[MAX_QUEUES]
        cdef int i
        for i in range(n_queues):
            assert isinstance(queues[i], cl.CommandQueue)
            queues_[i] = <cl_command_queue><voidptr_t>queues[i].obj_ptr
        errcheck(clAmdFftBakePlan(self.plan,
                                  n_queues, queues_,
                                  NULL, NULL))

    def enqueue_transform(self, 
                         queues, 
                         in_buffers, 
                         out_buffers = None,
                         direction_forward = True, 
                         wait_for_events = None, 
                         temp_buffer = None,
                         ):
        """Enqueue an FFT transform operation, and either return immediately, or block 
        waiting for events.

        This transform API is specific to the interleaved complex format, taking an input buffer with real
        and imaginary components paired together, and outputting the results into an output buffer in the
        same format.

        Parameters
        ----------
        queues     : list
                   of things

        in_buffers : array-like
                   array-like input data

        Other Parameters
        ----------------
        out_buffers : array-like, optional
                    if the plan is out-of-place, then we have out buffers

        direction_forward : bool, optional
                          this works like it sounds like it should

        wait_for_events : list, optional
                        I am not sure how this interface works

        temp_buffer : buffer, optional
                    I am not sure how this works

        Returns
        -------
            None

        Raises
        ------
            GpyFFT_Error
                An error occurred accessing the clAmdFftEnqueueTransform function

        Notes
        -----
            The underlying AMD FFT call is 'clAmdFftEnqueueTransform'
        """

        cdef int i

        cdef clAmdFftDirection direction
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
            queues_[i] = <cl_command_queue><voidptr_t>queue.obj_ptr
            
        cdef cl_event wait_for_events_array[MAX_WAITFOR_EVENTS]
        cdef cl_event* wait_for_events_ = NULL
        cdef n_waitfor_events = 0
        if wait_for_events is not None:
            n_waitfor_events = len(wait_for_events)
            assert n_waitfor_events <= MAX_WAITFOR_EVENTS
            for i, event in enumerate(wait_for_events):
                assert isinstance(event, cl.Event)
                wait_for_events_array[i] = <cl_event><voidptr_t>event.obj_ptr
            wait_for_events_ = &wait_for_events_array[0]

        cdef cl_mem in_buffers_[2]
        if isinstance(in_buffers, cl.Buffer):
            in_buffers = (in_buffers,)
        n_in_buffers = len(in_buffers)
        assert n_in_buffers <= 2
        for i, in_buffer in enumerate(in_buffers):
            assert isinstance(in_buffer, cl.Buffer)
            in_buffers_[i] = <cl_mem><voidptr_t>in_buffer.obj_ptr

        cdef cl_mem out_buffers_array[2]
        cdef cl_mem* out_buffers_ = NULL
        if out_buffers is not None:
            if isinstance(out_buffers, cl.Buffer):
                out_buffers = (out_buffers,)
            n_out_buffers = len(out_buffers)
            assert n_out_buffers in (1,2)
            for i, out_buffer in enumerate(out_buffers):
                assert isinstance(out_buffer, cl.Buffer)
                out_buffers_array[i] = <cl_mem><voidptr_t>out_buffer.obj_ptr
            out_buffers_ = &out_buffers_array[0]

        cdef cl_mem tmp_buffer_ = NULL
        if temp_buffer is not None:
            assert isinstance(temp_buffer, cl.Buffer)
            tmp_buffer_ = <cl_mem><voidptr_t>temp_buffer.obj_ptr

        cdef cl_event out_cl_events[MAX_QUEUES]

        errcheck(clAmdFftEnqueueTransform(self.plan,
                                          direction,
                                          n_queues,
                                          &queues_[0],
                                          n_waitfor_events,
                                          &wait_for_events_[0],
                                          out_cl_events,
                                          &in_buffers_[0],
                                          out_buffers_,
                                          tmp_buffer_))
        
        return tuple((cl.Event.from_cl_event_as_int(<long>out_cl_events[i]) for i in range(n_queues)))
            
        
            
        
        

#gpyfft = GpyFFT()


#cdef Plan PlanFactory():
    #cdef Plan instance = Plan.__new__(Ref)
    #instance.plan = None
    #return instance
