from __future__ import absolute_import, division, print_function

import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import gpyfft
import time

G = gpyfft.GpyFFT(debug=True)

print("clAmdFft Version: %d.%d.%d" % (G.get_version()))

context = cl.create_some_context()
queue = cl.CommandQueue(context)

print("context: 0x%x" % context.int_ptr)
print("queue: 0x%x" % queue.int_ptr)

shape = (256, 256, 256)

nd_data = np.random.random(shape).astype(np.complex64)
cl_data = cla.to_device(queue, nd_data)
cl_data_transformed = cla.empty_like(cl_data)

print("cl_data shape: %s" % str(cl_data.shape))

print('nd_data.shape/strides %s / %s ' % (nd_data.shape, nd_data.strides))
print('cl_data.shape/strides %s / %s' % (cl_data.shape, cl_data.strides))
print('cl_data_transformed.shape/strides %s / %s ' % (cl_data_transformed.shape, cl_data_transformed.strides))

plan = G.create_plan(context, cl_data.shape)
plan.strides_in = tuple(x // cl_data.dtype.itemsize for x in cl_data.strides)
plan.strides_out = tuple(x // cl_data.dtype.itemsize for x in cl_data_transformed.strides)


print('plan.strides_in %s' % str(plan.strides_in))
print('plan.strides_out %s' % str(plan.strides_out))
print('plan.distances %s' % str(plan.distances))
print('plan.batch_size %s' % str(plan.batch_size))

print('plan.inplace: %s' % plan.inplace)
plan.inplace = False
print('plan.inplace: %s' % plan.inplace)

print('plan.layouts: %s' % str(plan.layouts))

plan.precision = 1
print('plan.precision: %s' % plan.precision)

plan.scale_forward = 10
print('plan.scale_forward: %s' % plan.scale_forward)
plan.scale_forward = 1
print('plan.scale_forward: %s' % plan.scale_forward)

print('plan.transpose_result: %s' % plan.transpose_result)

t0 = time.time()
plan.bake(queue)
print("Bake time %.3fs" % (time.time() - t0))
print('plan.temp_array_size: %s' % plan.temp_array_size)

t1 = time.time()
plan.enqueue_transform((queue,),
                       (cl_data.data,),
                       (cl_data_transformed.data,)
                       )
queue.finish()
t2 = time.time()

t3 = time.time()
res = np.fft.fftn(nd_data)
t4 = time.time()
print("Error: %s" % abs(res - cl_data_transformed.get()).max())
print("Numpy %.3fs; OpenCL %.3fs; Speed_up: %.3f" % (t4 - t3, t2 - t1, (t4 - t3) / (t2 - t1)))
raw_input()
del plan
del G


print('nd_data:')
print(nd_data)
print('fft(nd_data):')
print(np.fft.fftn(nd_data))

del plan
del G


#raw_input()



