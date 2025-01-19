import streamlit as st
from diffusers import FluxPipeline
import torch
from PIL import Image
from huggingface_hub import login
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
import imageio
import os

# Загружаем модель при запуске приложения для FLUX
@st.cache_resource
def load_flux_pipeline():
    login("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")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16  # Используем смешанную точность
    )
    if torch.cuda.is_available():
        pipe.to("cuda")  # Переносим на GPU, если он доступен
    else:
        pipe.to("cpu")  # Иначе используем CPU
    return pipe

# Загружаем модель при запуске приложения для TRELLIS
@st.cache_resource
def load_trellis_model():
    login("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")
    model_name = "JeffreyXiang/TRELLIS-image-large"
    trellis_pipe = TrellisImageTo3DPipeline.from_pretrained(
        model_name
    )
    trellis_pipe.to("cuda")  # Отправляем модель на GPU
    return trellis_pipe

# Основное приложение
def main():
    st.title("Генерация изображений и 3D моделей")
    st.write("Используйте FLUX для создания изображений и TRELLIS для генерации 3D моделей.")

    # Выбор режима работы с уникальным ключом
    task = st.sidebar.selectbox(
        "Выберите задачу",
        ["Генерация изображения (FLUX)", "Создание 3D модели (TRELLIS)"],
        key="task_selector"  # Уникальный ключ
    )

    if task == "Генерация изображения (FLUX)":
        # Интерфейс для FLUX
        st.header("Генерация изображения")
        prompt = st.text_input("Введите текстовый запрос для генерации изображения:", key="flux_prompt")
        height = st.slider("Высота изображения (px):", 256, 1024, 512, step=64)
        width = st.slider("Ширина изображения (px):", 256, 1024, 512, step=64)
        guidance_scale = st.slider("Guidance Scale:", 1.0, 20.0, 7.5, step=0.5)
        num_inference_steps = st.slider("Количество шагов:", 10, 100, 50, step=10)
        if st.button("Сгенерировать", key="flux_generate"):
            if not prompt:
                st.warning("Пожалуйста, введите текстовый запрос.")
            else:
                with st.spinner("Генерация изображения..."):
                        flux_pipe = load_flux_pipeline()
                        image = flux_pipe(
                        prompt,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        ).images[0]
                        st.image(image, caption="Сгенерированное изображение", use_column_width=True)

    elif task == "Создание 3D модели (TRELLIS)":
        # Интерфейс для TRELLIS
        st.header("Создание 3D модели")
        uploaded_image = st.file_uploader("Загрузите изображение для преобразования в 3D", type=["png", "jpg", "jpeg"], key="trellis_uploader")
        if uploaded_image is not None:
            input_image = Image.open(uploaded_image).convert("RGB")
            st.image(input_image, caption="Загруженное изображение", use_column_width=True)

            if st.button("Создать 3D модель", key="trellis_generate"):
                with st.spinner("Создание 3D модели..."):
                    trellis_pipe = load_trellis_model()
                    # Генерация 3D модели через pipeline
                    outputs = trellis_pipe.run(input_image)
                    
                    # Извлекаем и отображаем 3D модель
                    # Рендерим гауссианы как пример (можно использовать другие результаты)
                    video = render_utils.render_video(outputs['gaussian'][0])['color']
                    video_path = "generated_3d_model_gaussian.mp4"
                    imageio.mimsave(video_path, video, fps=30)
                    st.video(video_path)
                    
                    # Преобразование 3D модели в формат GLB
                    glb = postprocessing_utils.to_glb(
                        outputs['gaussian'][0],
                        outputs['mesh'][0],
                        simplify=0.95,
                        texture_size=1024
                    )
                    glb_file_path = "generated_3d_model.glb"
                    glb.export(glb_file_path)

                    # Предоставляем кнопку для скачивания GLB
                    with open(glb_file_path, "rb") as f:
                        st.download_button("Скачать 3D модель (GLB)", data=f, file_name="generated_3d_model.glb", mime="application/octet-stream")

                    # Можно также предоставить скачивание PLY файлов
                    ply_file_path = "generated_3d_model.ply"
                    outputs['gaussian'][0].save_ply(ply_file_path)
                    with open(ply_file_path, "rb") as f:
                        st.download_button("Скачать 3D модель (PLY)", data=f, file_name="generated_3d_model.ply", mime="application/octet-stream")

# Запуск приложения
if __name__ == "__main__":
    main()
