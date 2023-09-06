import csv
import os
from collections import defaultdict
from datetime import datetime

import streamlit as st
import torch
import yaml

import defaults as dvar
import van_bot_components as vbc

col1, col2 = vbc.create_header(title="Vincent van Bot - produces your art in bulk")

ms = vbc.ModelSelection(col2)

sbc = vbc.SideBarConfigs(ms.width_val, ms.height_val, ms.num_inference_steps_val)

# Code related to bulk producing art
with st.container():
    is_portrait = st.checkbox("Generate portraits (prompt is about a person)", value=True)

    prompt = st.text_input("Describe the image you want to create (prompt)", help="Describe what picture you want",
                           value=ms.prompt_add)
    negative_prompt = st.text_input("Describe the aspects you **do not** want the image to have (negative prompt)",
                                    help="Describe what you do not want in the picture")

    # List from: https://docs.google.com/spreadsheets/d/1SRqJ7F_6yHVSOeCi3U82aA448TqEGrUlRrLLZ51abLg/edit#gid=1609688219
    with open('data/sd15_list_of_artists.csv') as f:
        reader = csv.DictReader(f)
        artist_tags = {}
        tags_artists = defaultdict(list)
        for r in reader:
            if r['Recognized by SD?'] == 'Yes':
                tags = r['Tags'].split(', ')
                artist_tags[r['Artists']] = tags
                for tag in tags:
                    tags_artists[tag].append(r['Artists'])

    selected_tags = st.multiselect(label='Artist Style (to select below)', options=tags_artists.keys(),
                                   default=['portrait'])
    # if any tag in the selection is a tag of the artist
    artist_options = [a for a in artist_tags if len(set(artist_tags[a]) & set(selected_tags)) > 0]

    selected_artists = st.multiselect(label='Artists to emulate their style', options=artist_options)

if st.button("Create"):
    # todo: disable all input but an interrupt

    list_gen_args = {
        artist: dict(prompt="{portrait} {prompt} in the style of {artist}, {tags}".format(
            portrait='Portrait of ' if is_portrait else '',
            prompt=prompt, artist=artist,
            tags=', '.join(artist_tags[artist])),
            negative_prompt=negative_prompt, width=sbc.width, height=sbc.height,
            num_inference_steps=sbc.num_inference_steps, num_images_per_prompt=sbc.num_images_per_prompt)

        for artist in selected_artists
    }

    progress_bar = st.progress(0, "Generating images...")

    base_fn = "{time}-{prompt}".format(time=datetime.now().strftime("%Y%m%d_%H%M"),
                                       prompt=prompt[:30].strip().replace(" ", "_"))
    if sbc.auto_save_image:
        os.makedirs(f"{dvar.output_path_many}/{base_fn}", exist_ok=True)

    total_artists = len(list_gen_args)
    curr_artist = 0.0
    for artist, gen_args in list_gen_args.items():
        progress_bar.progress(curr_artist / total_artists, f"Generating art in the style of {artist}")
        curr_artist += 1.0

        generator = torch.manual_seed(sbc.seed)
        result = ms.pipeline(**gen_args, generator=generator)

        gen_args['seed'] = sbc.seed
        gen_args['model_path'] = ms.model_path

        tot_images = len(result.images)

        st.text("Prompt: " + gen_args['prompt'])
        if gen_args['negative_prompt'] != "":
            st.text("Negative Prompt: " + gen_args['negative_prompt'])

        st.image(result.images, use_column_width=None)

        saved_str = ""
        if sbc.auto_save_image:
            fn = f"{dvar.output_path_many}/{base_fn}/{base_fn}_{artist}"
            with open(fn + '_meta.yml', 'wt') as fo:
                yaml.dump(gen_args, fo)

            for i, image in enumerate(result.images):
                add_num = '_{}'.format(i + 1) if tot_images > 1 else ''
                image.save(fn + add_num + ".png")
            saved_str = f" and saved on {dvar.output_path_many}/{base_fn}"

    progress_bar.progress(1.0, "All images created")
