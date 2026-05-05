import deeplake
# from openai import OpenAI
# import os

# Initialize the local dataset
ds = deeplake.empty('./openai_deeplake_db', overwrite=True)

# Create the tensors
ds.create_tensor('text', htype='text')
ds.create_tensor('metadata', htype='json')
ds.create_tensor('embedding', htype='embedding', dtype='float32', sample_compression=None)