import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

def noise(image):
    mean = 0.0
    var = 0.01
    sigma = var**0.5
    gauss = np.array(image.shape)
    gauss = np.random.normal(mean,sigma,image.shape)
    gauss = gauss.reshape(*image.shape)
    return image + torch.from_numpy(gauss).half().to("cuda")

def main():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    pipe = pipe.to("cuda")

    prompt = "a tiger dressed as a doctor"
    image = Image.new(mode="RGB", size=(512, 512), color=(128, 128, 128))
    with autocast("cuda"):
        for i in range(100):
            image = pipe(prompt, strength=0.25, guidance_scale=4)
            image = image.images[0]
            image.save(f"image_{i}.png")

if __name__ == "__main__":
    main()