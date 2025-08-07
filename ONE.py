import os
import torch
from PIL import Image, ImageFilter
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from GetAspectRatio import detect_monitor
from GetWeather import get_weather_info
from tqdm import tqdm

# ==== SETTINGS ====
TILE_SIZE = 1024
OVERLAP = 0.5
GUIDANCE = 7
STEPS = 40
SAVE_PROGRESS = True

# ==== LOAD MODELS ====
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading base model...")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    ).to(device)
    base_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(base_pipe.scheduler.config)

    print("Loading inpainting model...")
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype
    ).to(device)
    inpaint_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(inpaint_pipe.scheduler.config)

    return base_pipe, inpaint_pipe, device

# ==== MASK BUILDER ====
def make_soft_mask(width, height, direction):
    mask = Image.new("L", (width, height), 0)
    band_width = int(width * OVERLAP) if direction in ["right", "left"] else width
    band_height = int(height * OVERLAP) if direction in ["down"] else height
    band = Image.new("L", (band_width, band_height), 255).filter(ImageFilter.GaussianBlur(radius=64))

    if direction == "right":
        mask.paste(band, (width - band.width, 0))
    elif direction == "left":
        mask.paste(band, (0, 0))
    elif direction == "down":
        mask.paste(band, (0, height - band.height))
    return mask

# ==== RENDER TILE ====
def render_tile(pipe, canvas, prompt, x, y, direction):
    context_box = (x, y, x + TILE_SIZE, y + TILE_SIZE)
    context = canvas.crop(context_box)
    mask = make_soft_mask(TILE_SIZE, TILE_SIZE, direction)

    result = pipe(
        prompt=prompt,
        image=context,
        mask_image=mask,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        width=TILE_SIZE,
        height=TILE_SIZE
    ).images[0]

    canvas.paste(result, (x, y))
    return canvas

# ==== OUTPAINT FULL ====
def outpaint_full(pipe, canvas, prompt, width, height):
    step = int(TILE_SIZE * (1 - OVERLAP))

    for y in range(0, height, step):
        for x in range(0, width, step):
            if x == 0 and y == 0:
                continue
            direction = "right" if x > 0 else "down"
            canvas = render_tile(pipe, canvas, prompt, x, y, direction)

            if SAVE_PROGRESS:
                canvas.save("progress.png")
    return canvas

# ==== MAIN ====
def main():
    width, height = detect_monitor()
    city, country, desc, temp, tod = get_weather_info()
    prompt = f"A {desc} landscape in {city}, {country} during {tod}, temperature {temp}°C, ultra-realistic, cinematic lighting, consistent style, 8K masterpiece"

    print(f"Target Resolution: {width}x{height}")
    print("Prompt:", prompt)

    base_pipe, inpaint_pipe, device = load_models()

    # Generate Base Image with Stable Diffusion (not inpaint)
    base = base_pipe(prompt=prompt, guidance_scale=GUIDANCE, num_inference_steps=STEPS).images[0]
    base = base.resize((TILE_SIZE, TILE_SIZE))

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(base, (0, 0))
    canvas.save("canvas_start.png")

    # Outpaint
    final = outpaint_full(inpaint_pipe, canvas, prompt, width, height)
    final.save("wallpaper.png")
    print("✅ Wallpaper saved as wallpaper.png")

if __name__ == "__main__":
    main()
