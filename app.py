from diffusers import FluxPipeline
from huggingface_hub import login
import torch
# Авторизация на HuggingFace
login("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")

# Инициализация пайплайна на CPU
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float32  # Используем float32 для совместимости с CPU
)
pipe.to("cpu")  # Переносим весь процесс на CPU

# Устанавливаем параметры
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=512,  # Снизьте разрешение для уменьшения потребления памяти
    width=512,
    guidance_scale=3.5,
    num_inference_steps=30  # Уменьшение шагов для ускорения работы
).images[0]

# Сохранение изображения
image.save("flux-dev.png")

