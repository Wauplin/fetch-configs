import argparse
import glob
import json
import logging
import os
from hashlib import sha256

import diffusers
from accelerate import init_empty_weights
from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    ControlNetModel,
    Kandinsky3UNet,
    T2IAdapter,
    UNet2DConditionModel,
    UNet2DModel,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
    VQModel,
)
from huggingface_hub import HfApi
from huggingface_hub.utils import disable_progress_bars
from requests.exceptions import HTTPError
from tqdm import tqdm

# So we don't flood stdout with progress bars
disable_progress_bars()
# Ignore UNet config warnings
diffusers.utils.logging.set_verbosity(diffusers.logging.CRITICAL)

logging.basicConfig(
    filename="configs.log",
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    filemode="w",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--model_component", type=str, default="UNet")
parser.add_argument("--save_path", type=str, default="configs")

args = parser.parse_args()
os.environ["DIFFUSERS_VERBOSITY"] = "error"

SAVE_PATH = args.save_path
UNET_MODEL_TYPES = ["UNet2DModel", "UNet2DConditionModel", "Kandinsky3UNet", "UNet3DConditionModel", "UNetSpatioTemporalConditionModel", "ControlNetModel", "T2IAdapter"]
VAE_MODEL_TYPES = ["AutoencoderKL", "AsymmetricAutoencoderKL", "AutoencoderKLTemporalDecoder", "AutoencoderTiny", "ConsistencyDecoderVAE", "VQModel"]

api = HfApi()
models = api.list_models(library="diffusers", sort="downloads", direction=-1)

def download_model_configs(model_id, model_filename, save_path):
    # Gated models throw an error. This could be cleaner
    try:
        output = api.snapshot_download(model_id, allow_patterns=["config.json", "**/config.json", "model_index.json"], local_dir=save_path)

    except HTTPError as e:
        if not e.response.status_code == 403:
            logger.error(f"Could not download model config for {model_id}")

        return

    return output


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def filter_configs(config_paths, model_types):
    filtered_config_paths = []
    for config_path in config_paths:
        config = load_config(config_path)

        if "_class_name" not in config:
            continue

        if config["_class_name"] not in model_types:
            continue

        filtered_config_paths.append(config_path)

    return filtered_config_paths



def save_updated_config(model_metadata, config_path):
    org_id = model_metadata["org_id"]
    model_name = model_metadata["model_name"]

    config = load_config(config_path)
    save_directory = config_path.replace("config.json", "")

    model_cls_metadata = {}
    class_name = config["_class_name"]

    if not hasattr(diffusers, class_name):
        logger.info(f"Cannot import {class_name} in diffusers")
        return

    if class_name in UNET_MODEL_TYPES:
        model_type = "unet"
    elif class_name in VAE_MODEL_TYPES:
        model_type = "vae"
    else:
        model_type = "unknown"

    if class_name == "UNet2DConditionModel":
        model_cls = UNet2DConditionModel
    elif class_name == "UNet2DModel":
        model_cls = UNet2DModel
    elif class_name == "UNet3DConditionModel":
        model_cls = UNet3DConditionModel
    elif class_name == "UNetSpatioTemporalConditionModel":
        model_cls = UNetSpatioTemporalConditionModel
    elif class_name == "Kandinsky3UNet":
        model_cls = Kandinsky3UNet
    elif class_name == "ControlNetModel":
        model_cls = ControlNetModel
    elif class_name == "T2IAdapter":
        model_cls = T2IAdapter
    elif class_name == "AutoencoderKL":
        model_cls = AutoencoderKL
    elif class_name == "AsymmetricAutoencoderKL":
        model_cls = AsymmetricAutoencoderKL
    elif class_name == "AutoencoderKLTemporalDecoder":
        model_cls = AutoencoderKLTemporalDecoder
    elif class_name == "AutoencoderTiny":
        model_cls = AutoencoderTiny
    elif class_name == "ConsistencyDecoderVAE":
        model_cls = ConsistencyDecoderVAE
    elif class_name == "VQModel":
        model_cls = VQModel
    else:
        logger.info(f"Unknown model class {class_name}")
        return

    with init_empty_weights():
        try:
            model = model_cls.from_config(config_path)
            model.config_name = "config_updated.json"
            model.save_config(save_directory=save_directory)

            with open(f"{save_directory}/config_updated.json", "r") as fp:
                config_updated = json.load(fp)

            # hash the config to check how many configs are unique
            config_hash = sha256(str(json.dumps(config_updated)).encode()).hexdigest()
            del config_updated
            model_cls_metadata.update({"class_name": class_name, "config_hash": config_hash, "model_type": model_type})
            model_cls_metadata.update(model_metadata)

            with open(f"{save_directory}/metadata.json", "w") as fp:
                json.dump(model_cls_metadata, fp)

        except Exception:
            logger.error(f"Could not load UNet for {org_id}/{model_name}")
            return


def fetch_model_configs(model, cache):
    model_id = model.modelId
    if model_id in set(cache["fetched_models"]):
        return

    model_metadata = {}
    org_id,  model_name = model_id.split("/")
    model_metadata.update({"org_id": org_id,  "model_name": model_name, "downloads": model.downloads, "likes": model.likes})

    save_path = f"{SAVE_PATH}/{model_id}"
    config_path = download_model_configs(model_id, None, save_path)

    if config_path is None:
        logger.info(f"Could not fetch config for {model_id}")
        return

    all_configs = glob.glob(f"{config_path}/**/config.json", recursive=True)
    unet_config_paths = filter_configs(all_configs, UNET_MODEL_TYPES)

    for config_path in unet_config_paths:
        save_updated_config(model_metadata, config_path)

    vae_config_paths = filter_configs(all_configs, VAE_MODEL_TYPES)
    for config_path in vae_config_paths:
        save_updated_config(model_metadata, config_path)

    cache["fetched_models"].append(model_id)

    return "Success"


def main():
    if args.cache:
        with open(args.cache, "r") as fp:
            cache = json.load(fp)
    else:
        cache = {"fetched_models": []}

    try:
        # fetch all models from generator
        all_models = [model for model in models]
        for model in tqdm(all_models):
            fetch_model_configs(model, cache)

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupted")

    finally:
        with open("cache.json", "w") as fp:
            json.dump(cache, fp)

if __name__ == "__main__":
    main()
