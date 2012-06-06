cdef extern from "clAmdFft.h":
    ctypedef int cl_int
    ctypedef unsigned int cl_uint
    ctypedef unsigned long int cl_ulong
    ctypedef float cl_float
    ctypedef void* cl_context
    ctypedef void* cl_command_queue
    ctypedef void* cl_event
    ctypedef void* cl_mem


    ctypedef enum clAmdFftStatus:
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
        # status codes for clAmdFft
        CLFFT_BUGCHECK	#Bugcheck
        CLFFT_NOTIMPLEMENTED #Functionality is not implemented yet.
        CLFFT_TRANSPOSED_NOTIMPLEMENTED #Transposed functionality is not implemented for this transformation.
        CLFFT_FILE_NOT_FOUND #Tried to open an existing file on the host system, but failed.
        CLFFT_FILE_CREATE_FAILURE #Tried to create a file on the host system, but failed.
        CLFFT_VERSION_MISMATCH #Version conflict between client and library.
        CLFFT_INVALID_PLAN #Requested plan could not be found.
        CLFFT_DEVICE_NO_DOUBLE #Double precision not supported on this device.
        CLFFT_ENDSTATUS
        
    ctypedef enum clAmdFftDim:
        CLFFT_1D
        CLFFT_2D
        CLFFT_3D
        ENDDIMENSION

    ctypedef enum clAmdFftLayout:
        CLFFT_COMPLEX_INTERLEAVED
        CLFFT_COMPLEX_PLANAR
        CLFFT_HERMITIAN_INTERLEAVED
        CLFFT_HERMITIAN_PLANAR
        CLFFT_REAL
        ENDLAYOUT
        
    ctypedef enum clAmdFftPrecision:
        CLFFT_SINGLE
        CLFFT_DOUBLE
        CLFFT_SINGLE_FAST
        CLFFT_DOUBLE_FAST
        ENDPRECISION
        
    ctypedef enum clAmdFftDirection:
        CLFFT_FORWARD
        CLFFT_BACKWARD
        CLFFT_MINUS
        CLFFT_PLUS
        ENDDIRECTION

    ctypedef enum clAmdFftResultLocation:
        CLFFT_INPLACE
        CLFFT_OUTOFPLACE
        ENDPLACE

    ctypedef enum clAmdFftResultTransposed:
        CLFFT_NOTRANSPOSE
        CLFFT_TRANSPOSED
        ENDTRANSPOSED
        
    cdef enum:
        CLFFT_DUMP_PROGRAMS

    ctypedef struct clAmdFftSetupData:
        cl_uint major
        cl_uint minor
        cl_uint patch
        cl_ulong debugFlags

    ctypedef size_t clAmdFftPlanHandle

    clAmdFftStatus clAmdFftInitSetupData( clAmdFftSetupData* setupData )
    clAmdFftStatus clAmdFftSetup( clAmdFftSetupData* setupData )
    clAmdFftStatus clAmdFftTeardown( )
    clAmdFftStatus clAmdFftGetVersion( cl_uint* major, cl_uint* minor, cl_uint* patch )
    clAmdFftStatus clAmdFftCreateDefaultPlan( clAmdFftPlanHandle* plHandle, cl_context context, clAmdFftDim dim, size_t* clLengths )
    clAmdFftStatus clAmdFftCopyPlan( clAmdFftPlanHandle* out_plHandle, cl_context new_context, clAmdFftPlanHandle in_plHandle )
    clAmdFftStatus clAmdFftBakePlan( clAmdFftPlanHandle plHandle, cl_uint numQueues, cl_command_queue* commQueueFFT,
                                     void (*pfn_notify)(clAmdFftPlanHandle plHandle, void *user_data), void* user_data )
    clAmdFftStatus clAmdFftDestroyPlan( clAmdFftPlanHandle* plHandle )
    clAmdFftStatus clAmdFftGetPlanContext( clAmdFftPlanHandle plHandle, cl_context* context )
    clAmdFftStatus clAmdFftGetPlanPrecision( clAmdFftPlanHandle plHandle, clAmdFftPrecision* precision )
    clAmdFftStatus clAmdFftSetPlanPrecision( clAmdFftPlanHandle plHandle, clAmdFftPrecision precision )
    clAmdFftStatus clAmdFftGetPlanScale( clAmdFftPlanHandle plHandle, clAmdFftDirection dir, cl_float* scale )
    clAmdFftStatus clAmdFftSetPlanScale( clAmdFftPlanHandle plHandle, clAmdFftDirection dir, cl_float scale )
    clAmdFftStatus clAmdFftGetPlanBatchSize( clAmdFftPlanHandle plHandle, size_t* batchSize )
    clAmdFftStatus clAmdFftSetPlanBatchSize( clAmdFftPlanHandle plHandle, size_t batchSize )
    clAmdFftStatus clAmdFftGetPlanDim( clAmdFftPlanHandle plHandle, clAmdFftDim* dim, cl_uint* size )
    clAmdFftStatus clAmdFftSetPlanDim( clAmdFftPlanHandle plHandle, clAmdFftDim dim )
    clAmdFftStatus clAmdFftGetPlanLength( clAmdFftPlanHandle plHandle, clAmdFftDim dim, size_t* clLengths )
    clAmdFftStatus clAmdFftSetPlanLength( clAmdFftPlanHandle plHandle, clAmdFftDim dim, size_t* clLengths )
    clAmdFftStatus clAmdFftGetPlanInStride( clAmdFftPlanHandle plHandle, clAmdFftDim dim, size_t* clStrides )
    clAmdFftStatus clAmdFftSetPlanInStride( clAmdFftPlanHandle plHandle, clAmdFftDim dim, size_t* clStrides )
    clAmdFftStatus clAmdFftGetPlanOutStride( clAmdFftPlanHandle plHandle, clAmdFftDim dim, size_t* clStrides )
    clAmdFftStatus clAmdFftSetPlanOutStride( clAmdFftPlanHandle plHandle, clAmdFftDim dim, size_t* clStrides )
    clAmdFftStatus clAmdFftGetPlanDistance( clAmdFftPlanHandle plHandle, size_t* iDist, size_t* oDist )
    clAmdFftStatus clAmdFftSetPlanDistance( clAmdFftPlanHandle plHandle, size_t iDist, size_t oDist )
    clAmdFftStatus clAmdFftGetLayout( clAmdFftPlanHandle plHandle, clAmdFftLayout* iLayout, clAmdFftLayout* oLayout )
    clAmdFftStatus clAmdFftSetLayout( clAmdFftPlanHandle plHandle, clAmdFftLayout iLayout, clAmdFftLayout oLayout )
    clAmdFftStatus clAmdFftGetResultLocation( clAmdFftPlanHandle plHandle, clAmdFftResultLocation* placeness )
    clAmdFftStatus clAmdFftSetResultLocation( clAmdFftPlanHandle plHandle, clAmdFftResultLocation placeness )
    clAmdFftStatus clAmdFftGetPlanTransposeResult( clAmdFftPlanHandle plHandle, clAmdFftResultTransposed * transposed )
    clAmdFftStatus clAmdFftSetPlanTransposeResult( clAmdFftPlanHandle plHandle, clAmdFftResultTransposed transposed )
    clAmdFftStatus clAmdFftGetTmpBufSize( clAmdFftPlanHandle plHandle, size_t* buffersize )
    clAmdFftStatus clAmdFftEnqueueTransform(clAmdFftPlanHandle plHandle,
                                            clAmdFftDirection dir,
                                            cl_uint numQueuesAndEvents,
                                            cl_command_queue* commQueues,
                                            cl_uint numWaitEvents,
                                            cl_event* waitEvents,
                                            cl_event* outEvents,
                                            cl_mem* inputBuffers,
                                            cl_mem* outputBuffers,
                                            cl_mem tmpBuffer
                                            )
    

