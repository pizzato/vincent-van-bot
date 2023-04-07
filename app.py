import random
import os
import yaml
import uuid
import torch
from diffusers import StableDiffusionPipeline
import streamlit as st

run_event = False

models = {
    # "Family fined tuned model": dict(path="youruser/yourmodel",
    #                              width=512, height=512,
    #                              inference_steps=50,
    #                              prompt_add=["person1name", "person2name", "person3name", "person4name"]),
    "Dreamlike-Art 2.0": dict(path="dreamlike-art/dreamlike-photoreal-2.0",
                              width=768, height=768,
                              inference_steps=25,
                              prompt_add="photo"),
    "Analog Diffusion": dict(path="wavymulder/Analog-Diffusion",
                                   width=512, height=512,
                                   inference_steps=25,
                                   prompt_add="analog style,"),
    "PromptHero OpenJourney": dict(path="prompthero/openjourney-v4",
                                   width=512, height=512,
                                   inference_steps=25,
                                   prompt_add=""),
    "Arcane Diffusion": dict(path="nitrosocke/Arcane-Diffusion",
                                   width=512, height=512,
                                   inference_steps=25,
                                   prompt_add="arcane style,"),
    "Inkpunk Diffusion": dict(path="Envvi/Inkpunk-Diffusion",
                             width=512, height=512,
                             inference_steps=25,
                             prompt_add="nvinkpunk,"),

}

max_val = (1 << 53) - 1

picture_size_options = [256, 512, 768, 1024, 1280]

output_path = "output"

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

st.title('Creating images with AI')


@st.cache_resource
def load_pipeline(mpath):
    pipeline_ = StableDiffusionPipeline.from_pretrained(mpath, safety_checker=None)

    pipeline_ = pipeline_.to(device)
    if device == "mps":
        pipeline_.enable_attention_slicing()
        _ = pipeline_("test", num_inference_steps=1)

    return pipeline_


width_val = 512
height_val = 512
num_inference_steps_val = 25
prompt_add_options = ""

with st.expander("Model Selection"):
    if not st.checkbox('Input own model', value=False):
        model_name = st.selectbox("Model", options=models.keys(), index=0)

        model_path = models[model_name]['path']

        width_val = models[model_name]['width']
        height_val = models[model_name]['height']
        num_inference_steps_val = models[model_name]['inference_steps']
        prompt_add_options = models[model_name]['prompt_add']

        if type(prompt_add_options) == str:
            prompt_add = prompt_add_options
            prompt_add_str = prompt_add_options
        elif type(prompt_add_options) == list:
            prompt_add = prompt_add_options[0]
            prompt_add_str = ", ".join(prompt_add_options)

        st.text("Add to the prompt: {}".format(prompt_add_str))
    else:
        model_name = None
        model_path = st.text_input("Model name/path", value=list(models.values())[0]['path'])

with st.sidebar:
    pipeline = load_pipeline(model_path)
    st.text("Running on " + device)

    new_seed = st.checkbox('New random seed for every generation', value=True)
    if new_seed:
        seed = random.randint(1, max_val)
        st.text("Seed was: {}".format(seed))
    else:
        seed = 0

    seed = st.number_input("Seed", value=seed, min_value=0, max_value=max_val, key="seed")

    width = st.select_slider("Width", value=width_val, options=picture_size_options)
    height = st.select_slider("Height", value=height_val, options=picture_size_options)
    if device != 'mps':
        num_images_per_prompt = st.slider("Number of images to generate", value=1, min_value=1, max_value=100)
    else:
        num_images_per_prompt = 1

    num_inference_steps = st.slider("Number of inference steps", value=num_inference_steps_val, min_value=1, max_value=1000)

    auto_save_image = st.checkbox('Save image automatically', value=True)

with st.form("Image Generation"):
    prompt = st.text_input("Describe the image you want to create (prompt)", help="Describe what picture you want", value=prompt_add)
    negative_prompt = st.text_input("Describe the aspects you **do not** want the image to have (negative prompt)",
                                    help="Describe what you do not want in the picture")

    gen_args = dict(prompt=prompt, negative_prompt=negative_prompt,
                    width=width, height=height,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt)

    submitted = st.form_submit_button("Create")
    if submitted:

        # generator = torch.manual_seed(seed)

        with st.spinner("Generating image..."):
            result = pipeline(**gen_args)
            # result = pipeline(**gen_args, generator=generator)

        gen_args['seed'] = seed
        gen_args['model_path'] = model_path

        os.makedirs(output_path, exist_ok=True)
        base_fn = str(uuid.uuid4())
        tot_images = len(result.images)
        for i, image in enumerate(result.images):

            st.image(image)

            fn = output_path + '/' + base_fn

            with open(fn + '_meta.yml', 'wt') as fo:
                yaml.dump(gen_args, fo)

            if tot_images > 1:
                fn = fn + '_{}'.format(i + 1)

            image.save(fn + ".png")
