'''Tests before deploy'''

import random

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

torch.cuda.empty_cache()

# Fix for local running
seedValue=random.randint(0,4294967295)
torch.manual_seed(seedValue)

pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=torch.float16 # Fix for stable work
).to('cuda')

with autocast("cuda"):
    image = pipe(["Natasha"] * 2, guidance_scale=2)
    print(image.keys())
    print(image['nsfw_content_detected'])
    image = image['images'][0]
image.save("test.png")
