import streamlit as st
import torch
from diffusers import FluxPipeline
from huggingface_hub import login
login("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")
# Настройки
st.set_page_config(page_title="FLUX.1 [dev] Demo", page_icon=":guardsman:", layout="centered")

# Заголовок
st.title("FLUX.1 [dev] Text-to-Image Generation")

# Описание
st.write(
    "Это демо для генерации изображений по текстовому описанию с использованием модели FLUX.1 [dev]. Введите описание, и модель сгенерирует изображение!"
)

# Ввод текста
prompt = st.text_area("Введите описание для генерации изображения", "A cat holding a sign that says hello world")

# Параметры
height = st.slider("Высота изображения", min_value=256, max_value=1024, value=1024, step=256)
width = st.slider("Ширина изображения", min_value=256, max_value=1024, value=1024, step=256)
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
num_inference_steps = st.slider("Количество шагов инференса", min_value=1, max_value=100, value=50, step=1)

# Кнопка генерации
generate_button = st.button("Сгенерировать изображение")

if generate_button:
    with st.spinner("Загрузка модели и генерация изображения..."):
        try:
            # Загрузка модели и установка для CPU
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev", torch_dtype=torch.float32, low_cpu_mem_usage=False
            )
            pipe.enable_model_cpu_offload()  # Оптимизация для CPU, если не хватает VRAM

            # Генерация изображения
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]

            # Отображение изображения
            st.image(image, caption="Сгенерированное изображение", use_column_width=True)
            image.save("generated_image.png")
            st.success("Изображение успешно сгенерировано и сохранено!")

        except Exception as e:
            st.error(f"Произошла ошибка при генерации изображения: {e}")