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
    image = Image.open('debug/nerf-output-500.png').convert('RGB').resize((512, 512))
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=True
    )
    pipe = pipe.to("cuda")

    with autocast("cuda"):
        image = pipe("a DSLR photo of a tiger eating a bowl of cereal", init_image=image, strength=0.9, guidance_scale=10)
        image = image.images[0]
        image.save(f"image_0.png")

if __name__ == "__main__":
    main()