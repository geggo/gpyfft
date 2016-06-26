import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from gpyfft.fft import FFT

context = cl.create_some_context()
queue = cl.CommandQueue(context)

data_host = np.zeros((4, 1024, 1024), dtype = np.complex64)
#data_host[:] = some_useful_data
data_gpu = cla.to_device(queue, data_host)

transform = FFT(context, queue, (data_gpu,), axes = (2, 1))

event, = transform.enqueue()
event.wait()

result_host = data_gpu.get()
