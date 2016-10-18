import pyopencl as cl

def get_contexts():
    """
    Return list of OpenCL contexts for all (GPU) devices present in the system.
    """
    ALL_DEVICES = []
    for platform in cl.get_platforms():
        ALL_DEVICES += platform.get_devices(device_type = cl.device_type.GPU)
    contexts = [ cl.Context([device]) for device in ALL_DEVICES ]
    return contexts
