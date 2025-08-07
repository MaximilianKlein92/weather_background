import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler

def load_base_model(model_name="runwayml/stable-diffusion-v1-5"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using {device} with {dtype}")
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe, device, dtype

def load_inpaint_model(model_name="runwayml/stable-diffusion-inpainting", device="cuda", dtype=torch.float16):
    return StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
