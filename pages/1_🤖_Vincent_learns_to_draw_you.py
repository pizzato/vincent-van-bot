import json
import os
import random
from datetime import datetime

import streamlit as st
import yaml
from PIL import Image

import defaults as dvar
import van_bot_components as vbc
from extra_funcs import resize, run_process

col1, col2 = vbc.create_header(title="Vincent van Bot - learns how to draw you")

ms = vbc.ModelSelection(col2, init_pipeline=False)

with col2:
    new_model_name = st.text_input("Name of your model")

    resolution = st.select_slider("Training resolution (width and height)",
                                  value=ms.width_val, options=dvar.picture_size_options)

    nconcepts = st.slider("Number of concepts to add", min_value=1, max_value=10, value=1)

concepts = [("", dict())] * nconcepts

concept_tabs = st.tabs([f"Concept {n_ + 1}" for n_ in range(nconcepts)])

for nc in range(nconcepts):
    with concept_tabs[nc]:
        cb1, cb2 = st.columns([2, 2])
        cb21, cb22 = cb2.columns(2)

        cp_name = cb1.text_input("Concept name", key=f"concept_name_{nc}")

        cp_class = cb21.multiselect("Class of concept  being trained", key=f"concept_classes_{nc}",
                                    options=['man', 'woman', 'dog', 'cat', 'person', 'kid', 'boy', 'girl'])
        cp_class_custom = cb22.text_input("If needed specify class here",
                                          key=f"concept_class_custom_{nc}")
        cp_name_folder = cb1.text_input("Folder of concept photos (leave it blank to create) ",
                                        key=f"concept_name_folder_{nc}")
        cp_class_folder = cb2.text_input("Folder of class photos (leave it blank to create) ",
                                         key=f"concept_class_folder_{nc}")

        print("cp_class:", cp_class)
        print("cp_class_custom:", cp_class_custom)
        print("cp_class_folder:", cp_class_folder)
        _no_class_defined = False
        if cp_name == "":
            st.warning("Please ensure you have a concept name")
        else:
            if (cp_class == []) and (cp_class_custom == "") and (cp_class_folder == ""):
                st.warning("You it good to define a name for the class to have a defined folder")
                _no_class_defined = True

            if cp_name_folder == "":
                cp_name_folder = "{base}/{time}-{concept}".format(base=dvar.train_image_folder,
                                                                  time=datetime.now().strftime("%Y%m%d"),
                                                                  concept=cp_name)
            if cp_class_custom != "":
                cp_class.append(cp_class_custom)

            if _no_class_defined:
                cp_class = [cp_name]
                cp_class_folder = cp_name_folder
            else:
                if cp_class_folder == "":
                    cp_class_folder = "{base}/{time}-{concept}".format(base=dvar.train_image_folder,
                                                                       time=datetime.now().strftime("%Y%m%d"),
                                                                       concept='_'.join(cp_class))

            prompt_template = "photo of {}"
            _concept = dict(instance_prompt=prompt_template.format(cp_name),
                            class_prompt=prompt_template.format(', '.join(cp_class)),
                            instance_data_dir=cp_name_folder,
                            class_data_dir=cp_class_folder)

            concepts[nc] = (cp_name, _concept)
            print("concepts ", concepts)

concept_name_list = [cp_name for (cp_name, _) in concepts]
print("Concept List name:", concept_name_list)
if "" in concept_name_list:
    st.warning("Make sure to all information on concepts in the tab(s) above to continue")
    st.stop()

st.write("### Add images to concept")
cc1, cc2 = st.columns(2)
with cc1:
    with st.form('Image_Upload', clear_on_submit=True):
        cp_name = st.selectbox("Add image to which concept:", options=concept_name_list)
        upload_files = st.file_uploader("Images", type=['png', 'jpg'], accept_multiple_files=True)
        # cp_images = st.image(upload_files)

        upload_submit = st.form_submit_button("Upload")
        if upload_submit:
            _cp_name_folder = concepts[concept_name_list.index(cp_name)][1]['instance_data_dir']

            print('3 cp_name_folder: ', _cp_name_folder)
            if _cp_name_folder != "":
                os.makedirs(_cp_name_folder, exist_ok=True)
            upload_status = st.status(label="Uploading", state="running", expanded=False)
            for uploaded_file in upload_files:
                image_ = resize(Image.open(uploaded_file), resolution, resolution)

                image_.save(f"{_cp_name_folder}/{uploaded_file.name}")
                # with open(f"{_cp_name_folder}/{uploaded_file.name}","wb") as fo:
                #     fo.write(uploaded_file.getbuffer())
                upload_status.update(label=f"Recorded file in {_cp_name_folder}/{uploaded_file.name}")
            upload_status.update(label=f"Upload {len(upload_files)} files completed", state="complete")

with cc2:
    # Show how many images in concepts or show images
    with st.expander('Image preview in concepts', expanded=True):
        concept_preview_images = st.tabs([f"Concept {n_ + 1} - {concept_name_list[n_]}" for n_ in range(nconcepts)])

        for nc in range(nconcepts):
            with concept_preview_images[nc]:
                _cp_name_folder = concepts[nc][1]['instance_data_dir']
                _img_fn_list = []
                if os.path.isdir(_cp_name_folder):
                    for fn in os.listdir(_cp_name_folder):
                        fn_ = fn.lower()
                        if (fn_.endswith('jpg') or fn_.endswith('jpeg')) or fn_.endswith('png'):
                            _img_fn_list.append(f"{_cp_name_folder}/{fn}")

                    st.image(_img_fn_list, width=100)

st.markdown(""" ---
### Training
""")

with st.expander("Training parameters"):
    c21, c22, c23 = st.columns(3)
    with c21:
        train_batch_size = st.number_input("Train Batch Size", min_value=1, value=1, step=1,
                                           help="Batch size (per device) for the training dataloader")
        gradient_accumulation_steps = st.number_input("Gradient Accumulation Steps", min_value=1, value=1, step=1,
                                                      help="Number of updates steps to accumulate before performing a backward/update pass.")
        learning_rate = 5e-6  # 1e-6
        _scheduler_options = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                              "constant_with_warmup"]
        lr_scheduler = st.selectbox("Scheduler", options=_scheduler_options, index=_scheduler_options.index("constant"),
                                    help="The scheduler type to use.")
        lr_warmup_steps = st.number_input("LR schedule warm up steps", min_value=0, value=0, step=1,
                                          help="Number of steps for the warmup in the lr scheduler.")
    with c22:

        max_train_steps = st.number_input("Max training steps", min_value=100, value=10000, step=100,
                                          help="Total number of training steps to perform")

        pretrained_vae_name_or_path = st.text_input("Pretrained VAE name or path", value="stabilityai/sd-vae-ft-mse",
                                                    help="Pretrained VAE, don't change unless needed")

        revision = "fp16"
        with_prior_preservation = st.checkbox("With prior preservation loss", value=True,
                                              help="Flag to add prior preservation loss")
        if with_prior_preservation:
            prior_loss_weight = st.number_input("Prior loss weight", value=1.0,
                                                help="The weight of prior preservation loss")

        new_seed = st.checkbox('New random seed for every generation', value=True)
        if new_seed:
            seed = random.randint(1, dvar.max_val)
            st.text("Seed was: {}".format(seed))
        else:
            seed = 0

        train_text_encoder = st.checkbox("Train text encoder", value=True, help="Whether to train the text encoder")

        _mp_options = ["no", "fp16", "bf16"]
        mixed_precision = st.selectbox("Use mixed precision", options=_mp_options, index=_mp_options.index('fp16'),
                                       help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). " +
                                            "Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.  Default to the  " +
                                            " value of accelerate config of the current system or the flag passed with" +
                                            " the `accelerate.launch` command. Use this argument to override the " +
                                            "accelerate config.")

    with c23:

        use_8bit_adam = st.checkbox("Use 8-bit adam", value=True,
                                    help="Whether or not to use 8-bit Adam from bitsandbytes")

        num_class_images = st.number_input("Number of class images", value=30, min_value=0, step=5,
                                           help="Minimal class images for prior preservation loss."
                                                "If not have enough images, additional images will "
                                                "be sampled with class_prompt")

        sample_batch_size = st.number_input("Sample Batch Size", min_value=1, value=4, step=1,
                                            help="Batch size (per device) for sampling images")
        save_interval = st.number_input("Save interval", min_value=100, value=1000, step=10,
                                        help="Save weights every N steps")

        save_sample_prompt = '"{}"'.format(st.text_input("Save sample prompt", value="photo of " + concepts[0][0],
                                                         help="The prompt used to generate sample outputs to save"))



if st.button("Start training"):

    output_model_folder = f"{dvar.train_model_folder}/{new_model_name}"
    concepts_list_fn = f"{output_model_folder}/{dvar.concepts_list_default_fn}"
    os.makedirs(output_model_folder, exist_ok=True)

    concepts_list = [_concept for _, _concept in concepts]

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open(concepts_list_fn, "w") as fo:
        json.dump(concepts_list, fo, indent=4)

    params = dict(
        pretrained_model_name_or_path=ms.model_path,
        pretrained_vae_name_or_path=pretrained_vae_name_or_path,
        output_dir=output_model_folder,
        revision=revision,
        prior_loss_weight=prior_loss_weight,
        seed=seed,
        resolution=resolution,
        train_batch_size=train_batch_size,
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        num_class_images=num_class_images,
        sample_batch_size=sample_batch_size,
        max_train_steps=max_train_steps,
        save_interval=save_interval,
        save_sample_prompt=save_sample_prompt,
        concepts_list=concepts_list_fn
    )

    params_bool = dict(train_text_encoder=train_text_encoder,
                       with_prior_preservation=with_prior_preservation,
                       use_8bit_adam=use_8bit_adam)

    command_params = [f"--{pn}={params[pn]}" for pn in params] + [f"--{pn}" for pn in params_bool if params_bool[pn]]

    command = ["accelerate", "launch", "train_dreambooth.py"] + command_params
    print("Running:", ' '.join(["accelerate", "launch", "train_dreambooth.py"] + command_params))
    st.write("Running: " + ' '.join(["accelerate", "launch", "train_dreambooth.py"] + command_params))

    with st.status("Training: ", expanded=False, state="running") as status:
        for line_out in run_process(command):
            print(line_out)
            st.write(line_out)
            status.update(label=f"Training: {line_out}", state="running", expanded=False)

    status.update(label="Training completed", state="complete", expanded=False)

    # Add to the inference model list
    with open(dvar.model_config_file, 'rt') as fi:
        models = yaml.safe_load(fi)

    models[new_model_name] = dict(path=f"{output_model_folder}/{max_train_steps}",
                                  width=resolution, height=resolution,
                                  inference_steps=dvar.num_inference_steps_val,
                                  prompt_add=concept_name_list)

    with open(dvar.model_config_file, 'wt') as fo:
        yaml.safe_dump(models, fo)