import pyopencl as cl
import gpyfft

G = gpyfft.GpyFFT()

print "clAmdFft Version: %d.%d.%d"%(G.get_version())

context = cl.create_some_context()
queue = cl.CommandQueue(context)

print "context:", hex(context.obj_ptr)
print "queue:", hex(queue.obj_ptr)

plan = G.create_plan(context, (1024,))

print plan.get_precision()

plan.bake(queue.obj_ptr)

del plan


#raw_input()



