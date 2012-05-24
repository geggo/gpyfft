import pyopencl as cl
from gpyfft import gpyfft

context = cl.create_some_context()
queue = cl.CommandQueue(context)

