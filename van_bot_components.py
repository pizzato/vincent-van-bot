import torch
import random
from diffusers import StableDiffusionPipeline
import streamlit as st
import os
from auth import check_password
import yaml
import defaults as dvar

auth_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", True))  # Check token in secrets then env var

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


@st.cache_resource
def load_pipeline(mpath, safety_checker=False, use_auth_token=auth_token):
    if not safety_checker:
        pipeline_ = StableDiffusionPipeline.from_pretrained(mpath, safety_checker=None, use_auth_token=use_auth_token)
    else:
        pipeline_ = StableDiffusionPipeline.from_pretrained(mpath, use_auth_token=use_auth_token)

    pipeline_ = pipeline_.to(device)
    if device == "mps":
        pipeline_.enable_attention_slicing()
        _ = pipeline_("test", num_inference_steps=1)

    return pipeline_


def create_header(title="Vincent van Bot"):
    global auth_token
    st.set_page_config(page_title="Vincent van Bot - Create a model", layout="wide", initial_sidebar_state="auto",
                       menu_items=None, page_icon='images/VincentVanBot-Icon.png')

    st.title(title)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image('images/VincentVanBot.jpg')

    with col2:
        auth_token = st.secrets.get("HF_TOKEN", auth_token)  # Check token in secrets then env var

        if not check_password():
            st.stop()

    return (col1, col2)


class ModelSelection:
    def __init__(self, st_container, init_pipeline=True):

        with open(dvar.model_config_file, 'rt') as fi:
            models = yaml.safe_load(fi)

        with st_container.expander("Base Model Selection", expanded=True):
            col21, col22 = st.columns([1, 2])
            hf_auth_token_input = col22.text_input("HF Authentication Token", type="password",
                                                   help="Hugging Face Token if needed for private models")

            auth_token_to_use = auth_token if hf_auth_token_input == "" else hf_auth_token_input

            input_own_model = col21.checkbox('Input own model', value=False)
            safety_checker = col21.checkbox('Safety Checker', value=False)
            if not input_own_model:
                model_name = st.selectbox("Model", options=models.keys(), index=0)

                self.model_path = models[model_name].get('path', None)
                if self.model_path is None:
                    st.error('Model path cannot be undefined. Check model {} in yaml'.format(model_name))

                self.width_val = models[model_name].get('width', dvar.width_val)
                self.height_val = models[model_name].get('height', dvar.height_val)
                self.num_inference_steps_val = models[model_name].get('inference_steps', dvar.num_inference_steps_val)
                prompt_add_options = models[model_name].get('prompt_add', dvar.prompt_add_options)

                if type(prompt_add_options) == str:
                    self.prompt_add = prompt_add_options
                elif type(prompt_add_options) == list:
                    self.prompt_add = ", ".join(prompt_add_options)

                if self.prompt_add != "":
                    st.text("Add to the prompt: {}".format(self.prompt_add))
            else:
                model_name = None
                self.model_path = st.text_input("Model name/path", value=list(models.values())[0]['path'])
                self.width_val = dvar.width_val
                self.height_val =  dvar.height_val
                self.num_inference_steps_val = dvar.num_inference_steps_val
                self.prompt_add = ""

            self.pipeline = load_pipeline(self.model_path, safety_checker=safety_checker,
                                          use_auth_token=auth_token_to_use) \
                if init_pipeline else None

            st.text("Running on " + device)


class SideBarConfigs:
    def __init__(self, width_val, height_val, num_inference_steps_val):

        with st.sidebar:
            new_seed = st.checkbox('New random seed for every generation', value=True)
            if new_seed:
                seed = random.randint(1, dvar.max_val)
                st.text("Seed was: {}".format(seed))
            else:
                seed = 0

            self.seed = st.number_input("Seed", value=seed, min_value=0, max_value=dvar.max_val, key="seed")

            self.width = st.select_slider("Width", value=width_val, options=dvar.picture_size_options)
            self.height = st.select_slider("Height", value=height_val, options=dvar.picture_size_options)
            if device != 'mps':
                self.num_images_per_prompt = st.slider("Number of images to generate", value=4, min_value=1,
                                                       max_value=100)
            else:
                self.num_images_per_prompt = 1

            self.num_inference_steps = st.slider("Number of inference steps", value=num_inference_steps_val,
                                                 min_value=5,
                                                 max_value=250, step=5)

            self.draw_steps = st.slider("Steps to draw intermediate images",
                                        value=0, max_value=self.num_inference_steps // 2,
                                        help="Set to 0 for no intermediate images shown. "
                                             "Higher frequency will slow down process. Ignored on bullk art")

            self.auto_save_image = st.checkbox('Save image automatically', value=True)
