import time
import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import gpyfft

#NOTE: better benchmark contained in high level interface gpyfft/fft.py

G = gpyfft.GpyFFT()

print "clAmdFft Version: %d.%d.%d"%(G.get_version())

context = cl.create_some_context()
queue = cl.CommandQueue(context)

nd_data = np.ones((512, 512), dtype = np.complex64)
cl_data = cla.to_device(queue, nd_data)
cl_data_transformed = cla.empty_like(cl_data)

print 'data shape:', cl_data.shape

plan = G.create_plan(context, cl_data.shape)

plan.inplace = True #False
plan.precision = 1

print 'plan.inplace:', plan.inplace
print 'plan.precision:', plan.precision

plan.bake(queue)

def go(n_iter):
    for i in range(n_iter):
        plan.enqueue_transform((queue,), 
                               (cl_data.data,),
                               (cl_data_transformed.data,)
                               )
    queue.finish()


go(100)

nrun = 100
niter = 1
tic = time.time()
for k in range(nrun):
    go(niter)
toc = time.time()
print 'time per transform %.2f ms'%(1e3*(toc-tic)/(nrun*niter),)


#del plan
#del G




