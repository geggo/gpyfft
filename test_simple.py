import pyopencl as cl
from gpyfft import gpyfft

print "clAmdFft Version: %d.%d.%d"%(gpyfft.get_version())

context = cl.create_some_context()
queue = cl.CommandQueue(context)

plan = gpyfft.create_plan(context, (1024,))



