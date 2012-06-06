gpyfft class structure
**********************

.. autoclass:: gpyfft.GpyFFT
   :members:  get_version, create_plan

.. autoclass:: gpyfft.Plan
   :members:  __init__, precision, scale_forward, scale_backward, batch_size, get_dim, shape, strides_in, strides_out, distances, layouts, inplace, temp_array_size, transpose_result, bake, enqueue_transform
 
.. autoclass:: gpyfft.GpyFFT_Error

