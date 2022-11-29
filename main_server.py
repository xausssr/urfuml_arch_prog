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
if args.device == "cpu":
    dtype = torch.float
else:
    dtype = torch.float16

# Fix for local running
seedValue= random.randint(0,4294967295)
torch.manual_seed(seedValue)

@st.cache(allow_output_mutation=True)
def load():
    pipe = StableDiffusionPipeline.from_pretrained(
        'hakurei/waifu-diffusion',
        torch_dtype=dtype # Fix for stable work
    ).to(args.device)
    return pipe


def get_string():
    promt = st.text_input('Введите описание для генерации (на английском)')
    return promt


model = load()

st.title('Генерация аниме Streamlit')
prompt = get_string()
result = st.button('Сгенерировать изображение!')


if result:
    image = model([prompt], guidance_scale=7.5)['images']
    st.write('**Результаты распознавания:**')
    st.image(image[0], caption='Результат')
