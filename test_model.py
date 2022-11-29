'''Tests before deploy'''

import argparse
import random

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

parser = argparse.ArgumentParser()
parser.add_argument("--device", help="device for inference in torch notation (cpu, cuda)", default="cpu")

args = parser.parse_args()
print(f"Using {args.device} device for inference")

if "cuda" in args.device:
    torch.cuda.empty_cache()

# Fix for local running
seedValue=random.randint(0,4294967295)
torch.manual_seed(seedValue)

if args.device == "cpu":
    dtype = torch.float
else:
    dtype = torch.float16

pipe = StableDiffusionPipeline.from_pretrained(
    'hakurei/waifu-diffusion',
    torch_dtype=dtype # Fix for stable work
).to(args.device)

image = pipe(["Natasha"], guidance_scale=6)
print(image.keys())
print(image['nsfw_content_detected'])
image = image['images'][0]
image.save("test.png")
