import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import gpyfft

G = gpyfft.GpyFFT(debug=True)

print("clAmdFft Version: %d.%d.%d"%(G.get_version()))

context = cl.create_some_context()
queue = cl.CommandQueue(context)

print("context:", hex(context.int_ptr))
print("queue:", hex(queue.int_ptr))

nd_data = np.array([[1,2,3,4], [5,6,7,8]], dtype = np.complex64)
cl_data = cla.to_device(queue, nd_data)
cl_data_transformed = cla.empty_like(cl_data)

print("cl_data:")
print(cl_data)

print('nd_data.shape/strides', nd_data.shape, nd_data.strides)
print('cl_data.shape/strides', cl_data.shape, cl_data.strides)
print('cl_data_transformed.shape/strides', cl_data_transformed.shape, cl_data_transformed.strides)

plan = G.create_plan(context, cl_data.shape)
plan.strides_in = tuple(x//cl_data.dtype.itemsize for x in cl_data.strides)
plan.strides_out = tuple(x//cl_data.dtype.itemsize for x in cl_data_transformed.strides)


print('plan.strides_in', plan.strides_in)
print('plan.strides_out', plan.strides_out)
print('plan.distances', plan.distances)
print('plan.batch_size', plan.batch_size)

print('plan.inplace:', plan.inplace)
plan.inplace = False
print('plan.inplace:', plan.inplace)

print('plan.layouts:', plan.layouts)

plan.precision = 1
print('plan.precision:', plan.precision)

plan.scale_forward = 10
print('plan.scale_forward:', plan.scale_forward)
plan.scale_forward = 1
print('plan.scale_forward:', plan.scale_forward)

print('plan.transpose_result:', plan.transpose_result)

callback_kernel = """
float2 mulval(__global void* in,
              uint inoffset,
              __global void* userdata
              //__local void* localmem
)
{
float scalar = *((__global float*)userdata + inoffset);
float2 ret = *((__global float2*)in + inoffset) * scalar;
return ret;
}
"""

user_data = np.array([[1,2,3,4], [5,6,7,8]], dtype = np.float32)
user_data_device = cla.to_device(queue, user_data)

plan.set_callback('mulval', callback_kernel, 'pre',
                       user_data = user_data_device.data)

plan.bake(queue)
print('plan.temp_array_size:', plan.temp_array_size)

plan.enqueue_transform((queue,),
                       (cl_data.data,),
                       (cl_data_transformed.data,)
                       )
queue.finish()

print('cl_data:')
print(cl_data)
print('cl_data_transformed:')
print(cl_data_transformed)


print('nd_data:')
print(nd_data)
print('fft(nd_data):')
print(np.fft.fftn(nd_data))

del plan
del G


#raw_input()
