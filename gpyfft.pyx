# -*- coding: latin-1 -*-
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
    def __cinit__(self): #TODO: add debug flag
        cdef clAmdFftSetupData setup_data
        errcheck(clAmdFftInitSetupData(&setup_data))
        errcheck(clAmdFftSetup(&setup_data))

    def __dealloc__(self):
        errcheck(clAmdFftTeardown())

    def get_version(self):
        cdef cl_uint major, minor, patch
        errcheck(clAmdFftGetVersion(&major, &minor, &patch))
        return (major, minor, patch)
    
    def create_plan(self, context, tuple shape):
        return Plan(context, shape, self)
     
        
@cython.internal
cdef class Plan(object):

    cdef clAmdFftPlanHandle plan
    cdef object lib

    def __dealloc__(self):
        if self.plan:
            errcheck(clAmdFftDestroyPlan(&self.plan))
    
    def __cinit__(self):
        self.plan = 0

    def __init__(self, context, tuple shape, lib):
        self.lib = lib
        if not isinstance(context, cl.Context):
            raise TypeError('expected cl.Context as type of first argument')
        
        cdef cl_context context_handle = <cl_context><voidptr_t>context.obj_ptr

        ndim = len(shape)
        if ndim not in (1,2,3):
            raise ValueError('expected shape to be tuple of length 1,2 or 3')

        cdef size_t lengths[3]
        for i in range(ndim):
            lengths[i] = shape[i]
        
        clAmdFftCreateDefaultPlan(&self.plan, context_handle, ndim, &lengths[0])

    property precision:
        def __get__(self):
            cdef clAmdFftPrecision precision
            errcheck(clAmdFftGetPlanPrecision(self.plan, &precision))
            return precision
        def __set__(self, clAmdFftPrecision value):
            errcheck(clAmdFftSetPlanPrecision(self.plan, value))

    property scale_forward:
        def __get__(self):
            cdef cl_float scale
            errcheck(clAmdFftGetPlanScale(self.plan, CLFFT_FORWARD, &scale))
            return scale
        def __set__(self, cl_float value):
            errcheck(clAmdFftSetPlanScale(self.plan, CLFFT_FORWARD, value))

    property scale_backward:
        def __get__(self):
            cdef cl_float scale
            errcheck(clAmdFftGetPlanScale(self.plan, CLFFT_BACKWARD, &scale))
            return scale
        def __set__(self, cl_float value):
            errcheck(clAmdFftSetPlanScale(self.plan, CLFFT_BACKWARD, value))

    property inplace:
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

    def bake(self, queue):
        cdef cl_command_queue queue_handle = <cl_command_queue><voidptr_t>queue.obj_ptr
        errcheck(clAmdFftBakePlan(self.plan,
                                  1, &queue_handle,
                                  NULL, NULL))

    def enqueue_transform(self, 
                         queues, 
                         in_buffers, 
                         out_buffers = None,
                         direction_forward = True, 
                         wait_for_events = None, 
                         temp_buffers = None,
                         ):
        cdef clAmdFftDirection direction
        if direction_forward:
            direction = CLFFT_FORWARD
        else:
            direction = CLFFT_FORWARD
            
        cdef cl_command_queue queues_[MAX_QUEUES]
        if isinstance(queues, cl.CommandQueue):
            queues = (queues,)
        n_queues = len(queues)
        for i, queue in enumerate(queues):
            assert isinstance(queue, cl.CommandQueue)
            queues_[i] = <cl_command_queue><voidptr_t>queue.obj_ptr
            
        cdef cl_event wait_for_events_array[MAX_WAITFOR_EVENTS]
        cdef cl_event* wait_for_events_ = NULL
        cdef n_waitfor_events = 0
        if wait_for_events is not None:
            n_waitfor_events = len(wait_for_events)
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

        cdef cl_mem tmp_buffers_ = NULL #TODO: same as above

        errcheck(clAmdFftEnqueueTransform(self.plan,
                                          direction,
                                          n_queues,
                                          &queues_[0],
                                          n_waitfor_events,
                                          &wait_for_events_[0],
                                          NULL,
                                          &in_buffers_[0],
                                          out_buffers_,
                                          tmp_buffers_))

        
        
            
        
        

        
        
                





#gpyfft = GpyFFT()


#cdef Plan PlanFactory():
    #cdef Plan instance = Plan.__new__(Ref)
    #instance.plan = None
    #return instance
