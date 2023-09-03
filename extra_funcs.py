import subprocess

import torch
from PIL import Image

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)


# function lifted and inspired by https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/images.py
def resize(im, w, h):
    print(type(im))

    return im.resize((w, h), resample=LANCZOS)



# from: https://stackoverflow.com/a/4760274
def run_process(arguments):
    p = subprocess.Popen(arguments, text=True, universal_newlines=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    while True:
        # returns None while subprocess is running
        retcode = p.poll()
        line = p.stdout.readline()
        yield line
        if retcode is not None:
            break


# from https://stackoverflow.com/a/74373129
def image_from_latents(model, latents):
    # convert latents to image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = model.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        # convert to PIL Images
        image = model.numpy_to_pil(image)
        return [img for img in image]


