import logging
import json
import os
import argparse
from huggingface_hub import HfApi, ModelFilter, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

import diffusers
from diffusers import UNet2DConditionModel, UNet2DModel, UNet3DConditionModel, UNetSpatioTemporalConditionModel, Kandinsky3UNet
from requests.exceptions import HTTPError
from hashlib import sha256
from multiprocessing.pool import ThreadPool as Pool
from accelerate import init_empty_weights
from tqdm import tqdm

# So we don't flood stdout with progress bars
disable_progress_bars()
# Ignore UNet config warnings
diffusers.utils.logging.set_verbosity(diffusers.logging.CRITICAL)

logging.basicConfig(
    filename="fetch_configs.log",
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=str, default=None)
parser.add_argument("--model_component", type=str, default="unet")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_path", type=str, default="configs")

args = parser.parse_args()
os.environ["DIFFUSERS_VERBOSITY"] = "error"

SAVE_PATH = args.save_path
NUM_WORKERS = args.num_workers
MODEL_COMPONENT = args.model_component

api = HfApi()
filter = ModelFilter(library="diffusers")
models = api.list_models(filter=filter, sort="downloads", direction=-1)
pool = Pool(NUM_WORKERS)


def filter_model_component(model):
    if not model.siblings:
        return

    for sibling in model.siblings:
        if "config.json" not in sibling.rfilename:
            continue

        component_model_name = sibling.rfilename.split("/")[0]
        if MODEL_COMPONENT not in component_model_name:
            continue

        return sibling


def download_model_config(model_id, model_filename, save_path):
    # Gated models throw an error. This could be cleaner
    try:
        output = hf_hub_download(model_id, filename=model_filename, local_dir=save_path)
    except HTTPError as e:
        if not e.response.status_code == 403:
            logger.error(f"Could not download model config for {model_id}")

        return

    return output


def download_model_index(model_id, save_path):
    # Gated models throw an error. This could be cleaner
    try:
        output = hf_hub_download(model_id, filename="model_index.json", local_dir=save_path)
    except HTTPError as e:
        logger.error(f"Could not download model index for {model_id}")

        return

    return output


def fetch_model_configs(model, cache):
    model_id = model.modelId
    if model_id in set(cache["fetched_models"]):
        return

    component = filter_model_component(model)
    if not component:
        logger.info(f"No {MODEL_COMPONENT} found for {model_id}")
        return

    metadata = {}
    org_id,  model_name = model_id.split("/")
    metadata.update({"org_id": org_id,  "model_name": model_name, "downloads": model.downloads, "likes": model.likes})

    save_path = f"{SAVE_PATH}/{model_id}"
    config = download_model_config(model_id, component.rfilename, save_path)
    model_index = download_model_index(model_id, save_path)

    if config is None:
        logger.info(f"No {MODEL_COMPONENT} config found for {model_id}")
        return

    config_path = f"{save_path}/{MODEL_COMPONENT}/config.json"
    with open(config_path, "r") as fp:
        config = json.load(fp)

    class_name = config["_class_name"]
    save_directory = f"{save_path}/{MODEL_COMPONENT}"
    if class_name == "UNet2DConditionModel":
        unet_cls = UNet2DConditionModel
    elif class_name == "UNet2DModel":
        unet_cls = UNet2DModel
    elif class_name == "UNet3DConditionModel":
        unet_cls = UNet3DConditionModel
    elif class_name == "UNetSpatioTemporalConditionModel":
        unet_cls = UNetSpatioTemporalConditionModel
    elif class_name == "Kandinsky3UNet":
        unet_cls = Kandinsky3UNet
    else:
        logger.info(f"Unknown UNet class {class_name}")
        return

    with init_empty_weights():
        try:
            unet = unet_cls.from_config(config_path)
            unet.config_name = "config_updated.json"
            unet.save_config(save_directory=save_directory)

            with open(f"{save_directory}/config_updated.json", "r") as fp:
                config_updated = json.load(fp)

            # hash the config to check how many configs are unique
            config_hash = sha256(str(json.dumps(config_updated)).encode()).hexdigest()
            del config_updated
            metadata.update({"class_name": class_name, "config_hash": config_hash})

        except Exception as e:
            logger.error(f"Could not load UNet for {model_id}")
            return

    with open(f"{save_path}/{MODEL_COMPONENT}/metadata.json", "w") as fp:
        json.dump(metadata, fp)

    cache["fetched_models"].append(f"{model_id}-{MODEL_COMPONENT}")

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
            pool.apply(fetch_model_configs, args=(model, cache,))

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupted")

    finally:
        with open("cache.json", "w") as fp:
            json.dump(cache, fp)

        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
