import os
from datetime import datetime

import streamlit as st
import torch
import yaml

import defaults as dvar
import van_bot_components as vbc
from extra_funcs import image_from_latents

col1, col2 = vbc.create_header(title="Vincent van Bot - creates your art")

ms = vbc.ModelSelection(col2)

sbc = vbc.SideBarConfigs(ms.width_val, ms.height_val, ms.num_inference_steps_val)

with st.form("Image Generation"):
    prompt = st.text_input("Describe the image you want to create (prompt)", help="Describe what picture you want",
                           value=ms.prompt_add)
    negative_prompt = st.text_input("Describe the aspects you **do not** want the image to have (negative prompt)",
                                    help="Describe what you do not want in the picture")

    gen_args = dict(prompt=prompt, negative_prompt=negative_prompt,
                    width=sbc.width, height=sbc.height,
                    num_inference_steps=sbc.num_inference_steps,
                    num_images_per_prompt=sbc.num_images_per_prompt)

    submitted = st.form_submit_button("Create")

if submitted:

    generator = torch.manual_seed(sbc.seed)

    with (st.status("Generating image", expanded=False, state="running") as status):
        def update_status(s, _, latents):
            m_ = f"Step {s} of {sbc.num_inference_steps}"

            if (sbc.draw_steps > 0) and\
                    (((s > 0) and (s < sbc.num_inference_steps))
                        and (s % sbc.draw_steps == 0)):
                st.write(m_)
                inter_img = image_from_latents(ms.pipeline, latents)
                img_cols = st.columns(len(inter_img))
                for c_, im_ in zip(img_cols, inter_img):
                    c_.image(im_, use_column_width=True)

            status.update(label=f"Generating image: {m_}")

        result = ms.pipeline(**gen_args, generator=generator, callback=update_status, callback_steps=1)

        status.update(label="Generation completed", state="complete", expanded=False)

    gen_args['seed'] = sbc.seed
    gen_args['model_path'] = ms.model_path

    os.makedirs(dvar.output_path_single, exist_ok=True)
    base_fn = "{time}-{prompt}".format(time=datetime.now().strftime("%Y%m%d_%H%M"),
                                       prompt=prompt)
    tot_images = len(result.images)

    st.image(result.images, use_column_width=None)

    if sbc.auto_save_image:
        fn = dvar.output_path_single + '/' + base_fn
        with open(fn + '_meta.yml', 'wt') as fo:
            yaml.dump(gen_args, fo)

        for i, image in enumerate(result.images):
            if tot_images > 1:
                fn = fn + '_{}'.format(i + 1)
            image.save(fn + ".png")
