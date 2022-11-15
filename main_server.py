import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast


@st.cache(allow_output_mutation=True)
def load():
    pipe = StableDiffusionPipeline.from_pretrained(
        'hakurei/waifu-diffusion',
        torch_dtype=torch.float32
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
        image = model(prompt, guidance_scale=6)["sample"][0]
    image.save("test.png")
    st.write('**Результаты распознавания:**')
    st.image(image, caption='Результат')
