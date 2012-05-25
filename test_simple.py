import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import gpyfft

G = gpyfft.GpyFFT()

print "clAmdFft Version: %d.%d.%d"%(G.get_version())

context = cl.create_some_context()
queue = cl.CommandQueue(context)

print "context:", hex(context.obj_ptr)
print "queue:", hex(queue.obj_ptr)

nd_data = np.array([[1,0,1,0], [0,2,0,2]], dtype = np.complex64)
cl_data = cla.to_device(queue, nd_data.T)
cl_data_transformed = cla.empty_like(cl_data)

print "cl_data:"
print cl_data

plan = G.create_plan(context, cl_data.shape)

print 'plan.precision:', plan.precision

plan.scale_forward = 1
print 'plan.scale_forward:', plan.scale_forward
plan.scale_forward = 1
print 'plan.scale_forward:', plan.scale_forward

print 'plan.inplace:', plan.inplace
plan.inplace = False
print 'plan.inplace:', plan.inplace

plan.bake(queue)
plan.enqueue_transform((queue,), 
                       (cl_data.data,),
                       (cl_data_transformed.data,)
                       )
queue.finish()

print 'cl_data:'
print cl_data
print 'cl_data_transformed:'
print cl_data_transformed

print 'fft(nd_data):'
print np.fft.fftn(nd_data).T

del plan
del G


#raw_input()



