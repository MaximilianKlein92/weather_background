# FINAL OUTPAINTING ENGINE (High Quality, Snake Pattern, Fixes)

import torch
import time
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
from GetAspectRatio import detect_monitor
from GetWeather import get_weather_info
from tqdm import tqdm
import os

# ==== SETTINGS ====
TILE_SIZE = 1024
OVERLAP = 0.6  # 60% overlap for strong context
GUIDANCE = 7
STEPS = 40
SAVE_PROGRESS = True
SOFT_OVERLAP = True  # weiche Übergänge
SEED = int(time.time())  # dynamisch

# ==== INIT PIPELINE ====
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading models...")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype
    ).to(device)
    base_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(base_pipe.scheduler.config)

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=dtype
    ).to(device)
    inpaint_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(inpaint_pipe.scheduler.config)

    generator = torch.Generator(device=device).manual_seed(SEED)
    return base_pipe, inpaint_pipe, generator

# ==== WHITE-CHECK ====
def needs_render(region):
    hist = region.convert("L").histogram()
    white_pixels = hist[-1]
    return white_pixels > (region.size[0] * region.size[1] * 0.95)

# ==== CREATE SOFT MASK ====
def make_soft_mask(width, height):
    mask = Image.new("L", (width, height), 255)
    if SOFT_OVERLAP:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=32))
    return mask

# ==== CREATE CANVAS ====
def create_canvas(base_img, width, height):
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(base_img, (0, 0))
    return canvas

# ==== SNAKE OUTPAINT ====
def snake_outpaint(pipe, generator, canvas, prompt):
    width, height = canvas.size
    step = int(TILE_SIZE * (1 - OVERLAP))

    y = 0
    row = 0
    total_tiles = ((width - TILE_SIZE) // step + 1) * ((height - TILE_SIZE) // step + 1)
    pbar = tqdm(total=total_tiles, desc="Outpainting...")

    while y < height:
        x_range = range(0, width, step)
        if row % 2 != 0:
            x_range = reversed(x_range)

        for x in x_range:
            if x == 0 and y == 0:
                pbar.update(1)
                continue  # Basisbild überspringen

            # Begrenzungen prüfen
            box_x = min(x, width - TILE_SIZE)
            box_y = min(y, height - TILE_SIZE)
            crop_box = (box_x, box_y, box_x + TILE_SIZE, box_y + TILE_SIZE)

            region = canvas.crop(crop_box)
            if not needs_render(region):
                pbar.update(1)
                continue  # schon gefüllt

            # Kontext hinzufügen (größerer Ausschnitt um Stil zu wahren)
            context_left = max(0, box_x - int(TILE_SIZE * OVERLAP))
            context_top = max(0, box_y - int(TILE_SIZE * OVERLAP))
            context_right = min(width, box_x + TILE_SIZE + int(TILE_SIZE * OVERLAP))
            context_bottom = min(height, box_y + TILE_SIZE + int(TILE_SIZE * OVERLAP))
            context_box = (context_left, context_top, context_right, context_bottom)
            context_img = canvas.crop(context_box)

            # Maske nur für den eigentlichen Tile-Bereich
            mask = make_soft_mask(context_img.size[0], context_img.size[1])
            draw = ImageDraw.Draw(mask)
            draw.rectangle(
                (
                    box_x - context_left,
                    box_y - context_top,
                    box_x - context_left + TILE_SIZE,
                    box_y - context_top + TILE_SIZE,
                ),
                fill=255,
            )

            safe_width = (context_img.size[0] // 8) * 8
            safe_height = (context_img.size[1] // 8) * 8
            context_img = context_img.resize((safe_width, safe_height))
            mask = mask.resize((safe_width, safe_height))

            # Render
            result = pipe(
                prompt=prompt,
                image=context_img,
                mask_image=mask,
                guidance_scale=GUIDANCE,
                num_inference_steps=STEPS,
                width=safe_width,
                height=safe_height,
                generator=generator,
            ).images[0]

            # Rückeinbau in Canvas
            canvas.paste(result, (context_left, context_top))

            if SAVE_PROGRESS:
                canvas.save("progress.png")

            pbar.update(1)

        y += step
        row += 1

    pbar.close()
    return canvas

# ==== MAIN ====
def main():
    # Monitor-Auflösung
    target_width, target_height = detect_monitor()

    # Prompt aus Wetterdaten
    city, country, desc, temp, tod = get_weather_info()
    prompt = f"A {desc} landscape in {city}, {country} during {tod}, temperature {temp}°C, ultra-realistic, cinematic lighting, 8K masterpiece"
    print("Prompt:", prompt)

    base_pipe, inpaint_pipe, generator = load_models()

    # Basisbild
    print("Generating base image...")
    base = base_pipe(
        prompt=prompt,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        width=TILE_SIZE,
        height=TILE_SIZE,
        generator=generator,
    ).images[0]
    base.save("base.png")

    # Canvas vorbereiten
    canvas = create_canvas(base, target_width, target_height)
    canvas.save("canvas_start.png")

    # Outpainting
    final_image = snake_outpaint(inpaint_pipe, generator, canvas, prompt)
    final_image.save("wallpaper.png")
    print("✅ Wallpaper ready: wallpaper.png")

if __name__ == "__main__":
    main()
