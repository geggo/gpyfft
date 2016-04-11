cdef extern from "clFFT.h":
    ctypedef int cl_int
    ctypedef unsigned int cl_uint
    ctypedef unsigned long int cl_ulong
    ctypedef float cl_float

    ctypedef void* cl_context
    ctypedef void* cl_command_queue
    ctypedef void* cl_event
    ctypedef void* cl_mem

    # cdef struct _cl_context:
    #     pass
    # ctypedef _cl_context *cl_context

    # cdef struct _cl_command_queue:
    #     pass
    # ctypedef _cl_command_queue *cl_command_queue

    # cdef struct _cl_event:
    #     pass
    # ctypedef _cl_event *cl_event

    # cdef struct _cl_mem:
    #     pass
    # ctypedef _cl_mem *cl_mem


    enum:
        CLFFT_DUMP_PROGRAMS ##define constant

    cpdef enum clfftStatus_:
        CLFFT_INVALID_GLOBAL_WORK_SIZE
        CLFFT_INVALID_MIP_LEVEL
        CLFFT_INVALID_BUFFER_SIZE
        CLFFT_INVALID_GL_OBJECT
        CLFFT_INVALID_OPERATION
        CLFFT_INVALID_EVENT
        CLFFT_INVALID_EVENT_WAIT_LIST
        CLFFT_INVALID_GLOBAL_OFFSET
        CLFFT_INVALID_WORK_ITEM_SIZE
        CLFFT_INVALID_WORK_GROUP_SIZE
        CLFFT_INVALID_WORK_DIMENSION
        CLFFT_INVALID_KERNEL_ARGS
        CLFFT_INVALID_ARG_SIZE
        CLFFT_INVALID_ARG_VALUE
        CLFFT_INVALID_ARG_INDEX
        CLFFT_INVALID_KERNEL
        CLFFT_INVALID_KERNEL_DEFINITION
        CLFFT_INVALID_KERNEL_NAME
        CLFFT_INVALID_PROGRAM_EXECUTABLE
        CLFFT_INVALID_PROGRAM
        CLFFT_INVALID_BUILD_OPTIONS
        CLFFT_INVALID_BINARY
        CLFFT_INVALID_SAMPLER
        CLFFT_INVALID_IMAGE_SIZE
        CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR
        CLFFT_INVALID_MEM_OBJECT
        CLFFT_INVALID_HOST_PTR
        CLFFT_INVALID_COMMAND_QUEUE
        CLFFT_INVALID_QUEUE_PROPERTIES
        CLFFT_INVALID_CONTEXT
        CLFFT_INVALID_DEVICE
        CLFFT_INVALID_PLATFORM
        CLFFT_INVALID_DEVICE_TYPE
        CLFFT_INVALID_VALUE
        CLFFT_MAP_FAILURE
        CLFFT_BUILD_PROGRAM_FAILURE
        CLFFT_IMAGE_FORMAT_NOT_SUPPORTED
        CLFFT_IMAGE_FORMAT_MISMATCH
        CLFFT_MEM_COPY_OVERLAP
        CLFFT_PROFILING_INFO_NOT_AVAILABLE
        CLFFT_OUT_OF_HOST_MEMORY
        CLFFT_OUT_OF_RESOURCES
        CLFFT_MEM_OBJECT_ALLOCATION_FAILURE
        CLFFT_COMPILER_NOT_AVAILABLE
        CLFFT_DEVICE_NOT_AVAILABLE
        CLFFT_DEVICE_NOT_FOUND
        CLFFT_SUCCESS
        CLFFT_BUGCHECK
        CLFFT_NOTIMPLEMENTED
        CLFFT_TRANSPOSED_NOTIMPLEMENTED
        CLFFT_FILE_NOT_FOUND
        CLFFT_FILE_CREATE_FAILURE
        CLFFT_VERSION_MISMATCH
        CLFFT_INVALID_PLAN
        CLFFT_DEVICE_NO_DOUBLE
        CLFFT_DEVICE_MISMATCH

    ctypedef clfftStatus_ clfftStatus

    cpdef enum clfftDim_:
        CLFFT_1D
        CLFFT_2D
        CLFFT_3D

    ctypedef clfftDim_ clfftDim

    cpdef enum clfftLayout_:
        CLFFT_COMPLEX_INTERLEAVED
        CLFFT_COMPLEX_PLANAR
        CLFFT_HERMITIAN_INTERLEAVED
        CLFFT_HERMITIAN_PLANAR
        CLFFT_REAL

    ctypedef clfftLayout_ clfftLayout

    cpdef enum clfftPrecision_:
        CLFFT_SINGLE
        CLFFT_DOUBLE
        CLFFT_SINGLE_FAST
        CLFFT_DOUBLE_FAST

    ctypedef clfftPrecision_ clfftPrecision

    cpdef enum clfftDirection_:
        CLFFT_FORWARD
        CLFFT_BACKWARD
        CLFFT_MINUS
        CLFFT_PLUS

    ctypedef clfftDirection_ clfftDirection

    cpdef enum clfftResultLocation_:
        CLFFT_INPLACE
        CLFFT_OUTOFPLACE

    ctypedef clfftResultLocation_ clfftResultLocation

    cpdef enum clfftResultTransposed_:
        CLFFT_NOTRANSPOSE
        CLFFT_TRANSPOSED

    ctypedef clfftResultTransposed_ clfftResultTransposed

    cdef struct clfftSetupData_:
        cl_uint major
        cl_uint minor
        cl_uint patch
        cl_ulong debugFlags

    ctypedef clfftSetupData_ clfftSetupData

    cpdef enum clfftCallbackType_:
        PRECALLBACK
        POSTCALLBACK

    ctypedef clfftCallbackType_ clfftCallbackType

    ctypedef size_t clfftPlanHandle

    
    clfftStatus clfftInitSetupData(clfftSetupData *setupData)
    clfftStatus clfftSetup(const clfftSetupData *setupData)
    clfftStatus clfftTeardown()
    clfftStatus clfftGetVersion(cl_uint *major, cl_uint *minor, cl_uint *patch)
    clfftStatus clfftCreateDefaultPlan(clfftPlanHandle *plHandle, cl_context context,
                                       #const clfftDim dim, 
                                       clfftDim dim, 
                                       const size_t *clLengths)
    clfftStatus clfftCopyPlan(clfftPlanHandle *out_plHandle, cl_context new_context, clfftPlanHandle in_plHandle)

    clfftStatus clfftBakePlan(clfftPlanHandle plHandle, 
                              cl_uint numQueues, 
                              cl_command_queue *commQueueFFT, 
                              #void (*pfn_notify)(unsigned long, void *),
                              void (*pfn_notify)(clfftPlanHandle plHandle, void *user_data),
                              void *user_data)
    
    clfftStatus clfftDestroyPlan(clfftPlanHandle *plHandle)
    clfftStatus clfftGetPlanContext(const clfftPlanHandle plHandle, cl_context *context)
    clfftStatus clfftGetPlanPrecision(const clfftPlanHandle plHandle, clfftPrecision *precision)
    clfftStatus clfftSetPlanPrecision(clfftPlanHandle plHandle, clfftPrecision precision)
    clfftStatus clfftGetPlanScale(const clfftPlanHandle plHandle, clfftDirection dir, cl_float *scale)
    clfftStatus clfftSetPlanScale(clfftPlanHandle plHandle, clfftDirection dir, cl_float scale)
    clfftStatus clfftGetPlanBatchSize(const clfftPlanHandle plHandle, size_t *batchSize)
    clfftStatus clfftSetPlanBatchSize(clfftPlanHandle plHandle, size_t batchSize)
    clfftStatus clfftGetPlanDim(const clfftPlanHandle plHandle, clfftDim *dim, cl_uint *size)
    clfftStatus clfftSetPlanDim(clfftPlanHandle plHandle, const clfftDim dim)
    clfftStatus clfftGetPlanLength(const clfftPlanHandle plHandle, const clfftDim dim, size_t *clLengths)
    clfftStatus clfftSetPlanLength(clfftPlanHandle plHandle, const clfftDim dim, const size_t *clLengths)
    clfftStatus clfftGetPlanInStride(const clfftPlanHandle plHandle, const clfftDim dim, size_t *clStrides)
    clfftStatus clfftSetPlanInStride(clfftPlanHandle plHandle, const clfftDim dim, size_t *clStrides)
    clfftStatus clfftGetPlanOutStride(const clfftPlanHandle plHandle, const clfftDim dim, size_t *clStrides)
    clfftStatus clfftSetPlanOutStride(clfftPlanHandle plHandle, const clfftDim dim, size_t *clStrides)
    clfftStatus clfftGetPlanDistance(const clfftPlanHandle plHandle, size_t *iDist, size_t *oDist)
    clfftStatus clfftSetPlanDistance(clfftPlanHandle plHandle, size_t iDist, size_t oDist)
    clfftStatus clfftGetLayout(const clfftPlanHandle plHandle, clfftLayout *iLayout, clfftLayout *oLayout)
    clfftStatus clfftSetLayout(clfftPlanHandle plHandle, clfftLayout iLayout, clfftLayout oLayout)
    clfftStatus clfftGetResultLocation(const clfftPlanHandle plHandle, clfftResultLocation *placeness)
    clfftStatus clfftSetResultLocation(clfftPlanHandle plHandle, clfftResultLocation placeness)
    clfftStatus clfftGetPlanTransposeResult(const clfftPlanHandle plHandle, clfftResultTransposed *transposed)
    clfftStatus clfftSetPlanTransposeResult(clfftPlanHandle plHandle, clfftResultTransposed transposed)
    clfftStatus clfftGetTmpBufSize(const clfftPlanHandle plHandle, size_t *buffersize)
    clfftStatus clfftSetPlanCallback(clfftPlanHandle plHandle,
                                     const char* funcName,
                                     const char* funcString,
                                     int localMemSize,
                                     clfftCallbackType callbackType,
                                     cl_mem *userdata,
                                     int numUserdataBuffers)
    clfftStatus clfftEnqueueTransform(clfftPlanHandle plHandle,
                                      clfftDirection dir, 
                                      cl_uint numQueuesAndEvents, 
                                      cl_command_queue *commQueues, 
                                      cl_uint numWaitEvents, 
                                      const cl_event *waitEvents, 
                                      cl_event *outEvents, 
                                      cl_mem *inputBuffers, 
                                      cl_mem *outputBuffers, 
                                      cl_mem tmpBuffer
                                      )
    

    cdef enum _clErrorCodes:
        CL_SUCCESS
        CL_DEVICE_NOT_FOUND
        CL_DEVICE_NOT_AVAILABLE
        CL_COMPILER_NOT_AVAILABLE
        CL_MEM_OBJECT_ALLOCATION_FAILURE
        CL_OUT_OF_RESOURCES
        CL_OUT_OF_HOST_MEMORY
        CL_PROFILING_INFO_NOT_AVAILABLE
        CL_MEM_COPY_OVERLAP
        CL_IMAGE_FORMAT_MISMATCH
        CL_IMAGE_FORMAT_NOT_SUPPORTED
        CL_BUILD_PROGRAM_FAILURE
        CL_MAP_FAILURE
        CL_MISALIGNED_SUB_BUFFER_OFFSET
        CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
        CL_COMPILE_PROGRAM_FAILURE
        CL_LINKER_NOT_AVAILABLE
        CL_LINK_PROGRAM_FAILURE
        CL_DEVICE_PARTITION_FAILED
        CL_KERNEL_ARG_INFO_NOT_AVAILABLE
        CL_INVALID_VALUE
        CL_INVALID_DEVICE_TYPE
        CL_INVALID_PLATFORM
        CL_INVALID_DEVICE
        CL_INVALID_CONTEXT
        CL_INVALID_QUEUE_PROPERTIES
        CL_INVALID_COMMAND_QUEUE
        CL_INVALID_HOST_PTR
        CL_INVALID_MEM_OBJECT
        CL_INVALID_IMAGE_FORMAT_DESCRIPTOR
        CL_INVALID_IMAGE_SIZE
        CL_INVALID_SAMPLER
        CL_INVALID_BINARY
        CL_INVALID_BUILD_OPTIONS
        CL_INVALID_PROGRAM
        CL_INVALID_PROGRAM_EXECUTABLE
        CL_INVALID_KERNEL_NAME
        CL_INVALID_KERNEL_DEFINITION
        CL_INVALID_KERNEL
        CL_INVALID_ARG_INDEX
        CL_INVALID_ARG_VALUE
        CL_INVALID_ARG_SIZE
        CL_INVALID_KERNEL_ARGS
        CL_INVALID_WORK_DIMENSION
        CL_INVALID_WORK_GROUP_SIZE
        CL_INVALID_WORK_ITEM_SIZE
        CL_INVALID_GLOBAL_OFFSET
        CL_INVALID_EVENT_WAIT_LIST
        CL_INVALID_EVENT
        CL_INVALID_OPERATION
        CL_INVALID_GL_OBJECT
        CL_INVALID_BUFFER_SIZE
        CL_INVALID_MIP_LEVEL
        CL_INVALID_GLOBAL_WORK_SIZE
        CL_INVALID_PROPERTY
        CL_INVALID_IMAGE_DESCRIPTOR
        CL_INVALID_COMPILER_OPTIONS
        CL_INVALID_LINKER_OPTIONS
        CL_INVALID_DEVICE_PARTITION_COUNT
        CL_INVALID_PIPE_SIZE
        CL_INVALID_DEVICE_QUEUE

    cdef enum cl_mem_flags:
        CL_MEM_READ_WRITE 
        CL_MEM_WRITE_ONLY 
        CL_MEM_READ_ONLY 
        CL_MEM_USE_HOST_PTR 
        CL_MEM_ALLOC_HOST_PTR 
        CL_MEM_COPY_HOST_PTR
    cdef enum cl_buffer_create_type:
        CL_BUFFER_CREATE_TYPE_REGION
    cdef enum cl_mem_info:
        CL_MEM_TYPE  
        CL_MEM_FLAGS 
        CL_MEM_SIZE 
        CL_MEM_HOST_PTR 
        CL_MEM_MAP_COUNT 
        CL_MEM_REFERENCE_COUNT 
        CL_MEM_CONTEXT
    cl_mem clCreateSubBuffer(cl_mem buffer,
                             int flags,
                             cl_buffer_create_type buffer_create_type,
                             const void *buffer_create_info,
                             cl_int *errcode_ret)
    cl_int clReleaseMemObject(cl_mem memobj)
    cl_int clGetMemObjectInfo(cl_mem memobj,
                              cl_mem_info param_name,
                              size_t param_value_size,
                              void *param_value,
                              size_t *param_value_size_ret)

cdef struct _cl_buffer_region:
    size_t origin
    size_t size
