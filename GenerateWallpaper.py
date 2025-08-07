from GetAspectRatio import detect_monitor
from GetWeather import build_prompt
from ModelLoader import load_base_model, load_inpaint_model
from OutpaintingHelper import create_canvas

def main():
    # Detect monitor
    width, height = detect_monitor()

    # Load models
    pipe, device, dtype = load_base_model()
    inpaint_pipe = load_inpaint_model(device=device, dtype=dtype)

    # Build weather-based prompt
    prompt = build_prompt()
    print("Prompt:", prompt)

    # Generate base image
    base_image = pipe(prompt, guidance_scale=12, num_inference_steps=40).images[0]
    base_image.save("base_image.png")

    # Prepare canvas for outpainting
    canvas, mask = create_canvas(base_image, width, height)

    # Outpaint
    final_image = inpaint_pipe(prompt=prompt, 
                               image=canvas, 
                               mask_image=mask, 
                               guidance_scale=6, 
                               num_inference_steps=20).images[0]
    
    final_image.save("wallpaper.png")
    print("Wallpaper saved as wallpaper.png")

if __name__ == "__main__":

    main()
