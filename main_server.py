import random

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

# Fix for local running
seedValue=random.randint(0,4294967295)
torch.manual_seed(seedValue)

@st.cache(allow_output_mutation=True)
def load():
    pipe = StableDiffusionPipeline.from_pretrained(
        'hakurei/waifu-diffusion',
        torch_dtype=torch.float16 # Fix for stable work
    ).to('cuda')
    return pipe


def get_string():
    promt = st.text_input('Введите описание для генерации (на английском)')
    return promt


model = load()

st.title('Генерация аниме Streamlit')
prompt = get_string()
result = st.button('Сгенерировать изображение!')


if result:
    with autocast("cuda"):
        image = model([prompt] * 2, guidance_scale=2)['images']
    st.write('**Результаты распознавания:**')
    st.image(image[0], caption='Результат №1')
    st.image(image[1], caption='Результат №2')
