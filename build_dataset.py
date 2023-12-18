import argparse
import glob
import inspect
import json
import logging
from hashlib import sha256

import diffusers
import pandas as pd
from huggingface_hub import HfApi

logging.basicConfig(
    filename="build.log",
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    filemode="w",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

api = HfApi()

filter_unet_params = ["block_out_channels", "cross_attention_dim", "sample_size", "attention_head_dim", "in_channels"]

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=".")
parser.add_argument("--repo_id", type=str, default=None)

args = parser.parse_args()


def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config

    except Exception:
        logger.info(f"Failed to load config at {config_path}")
        return


def align_updated_configs_with_model_index(config_df, model_index_df):
    """Some custom models/pipelines have modified the UNet class so that it can't be loaded directly
    by diffusers UNet classes.

    This function creates a dataframes that aligns the updated configs with the model index so that
    we only have data for pipelines that use diffusers native model classes

    """
    df = config_df.merge(model_index_df, on=["org_id", "model_name"], how="left")
    return df


def make_model_index_df(model_index_files):
    model_index_configs = []
    for model_index_file in model_index_files:
        conf = load_config(model_index_file)
        if not conf:
            continue
        if "_class_name" not in conf:
            continue

        conf.update({"model_index_path": model_index_file})
        model_index_configs.append(conf)

    data = []
    for conf in model_index_configs:
        model_index_path = conf["model_index_path"]
        org_id = model_index_path.split("/")[-3]
        model_name = model_index_path.split("/")[-2]
        pipeline = conf["_class_name"]

        data.append({"org_id": org_id, "model_name": model_name, "pipeline": pipeline})

    model_index_df = pd.DataFrame(data)
    return model_index_df


def compute_config_hash(config_path):
    with open(config_path, "r") as fp:
        config = json.load(fp)

    if "_class_name" not in config:
        return None

    cls = getattr(diffusers, config["_class_name"])
    cls_params = inspect.signature(cls).parameters

    new_config = {}
    new_config.update(config)
    for param in config.keys():
        if param not in cls_params:
            new_config.pop(param, None)

    for param in filter_unet_params:
        if param in new_config:
            new_config.pop(param, None)

    config_hash = sha256(str(json.dumps(new_config)).encode()).hexdigest()
    return config_hash


def main():
    input_path = args.input_path

    model_index_files = sorted(glob.glob(f"{input_path}/**/model_index.json", recursive=True))
    model_index_df = make_model_index_df(model_index_files)

    metadata_files = sorted(glob.glob(f"{input_path}/**/metadata.json", recursive=True))
    metadata_configs = [load_config(metadata_file) for metadata_file in metadata_files]

    df = pd.DataFrame(metadata_configs)

    updated_config_files = sorted(glob.glob(f"{input_path}/**/config_updated.json", recursive=True))
    df["updated_config_path"] = updated_config_files
    df["config_hash"] = df["updated_config_path"].apply(lambda x: compute_config_hash(x))
    df = align_updated_configs_with_model_index(df, model_index_df)

    metadata_path = f"{args.output_path}/metadata.csv"

    df.to_csv(metadata_path)
    if not args.repo_id:
        return

    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=metadata_path,
        path_in_repo="metadata.csv",
        commit_message="add metadata file"
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=input_path,
        path_in_repo="configs",
        ignore_patterns=["**/metadata.json"],
    )



if __name__ == "__main__":
    main()