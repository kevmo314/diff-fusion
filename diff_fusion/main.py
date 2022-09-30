import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
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

    prompt = "a zoomed out DSLR photo of a table with dim sum on it"
    with autocast("cuda"):
        image = pipe(prompt)
        image = image.images[0]
        image.save(f"image_0.png")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16,
        use_auth_token=True
    )
    pipe = pipe.to("cuda")

    with autocast("cuda"):
        for i in range(100):
            image = pipe(prompt, init_image=image, strength=0.25, guidance_scale=4)
            image = image.images[0]
            image.save(f"image_{i + 1}.png")

if __name__ == "__main__":
    main()