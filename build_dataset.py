import argparse
import glob
import json

import pandas as pd
from huggingface_hub import HfApi

api = HfApi()

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=".")
parser.add_argument("--repo_id", type=str, default=None)

args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def align_updated_configs_with_model_index(config_df, model_index_df):
    """Some custom models/pipelines have modified the UNet class so that it can't be loaded directly
    by diffusers UNet classes.

    This function creates a dataframes that aligns the updated configs with the model index so that
    we only have data for pipelines that use diffusers native model classes

    """
    df = config_df.merge(model_index_df, on=["org_id", "model_name"], how="left")
    return df


def make_model_index_df(model_index_files):
    model_index_configs = [load_config(model_index_file) for model_index_file in model_index_files]
    pipelines = [conf["_class_name"] for conf in model_index_configs]
    data = [{"org_id": midx.split("/")[1], "model_name": midx.split("/")[2], "pipeline": pipeline} for midx, pipeline in zip(model_index_files, pipelines)]
    model_index_df = pd.DataFrame(data)

    return model_index_df


def main():
    input_path = args.input_path

    model_index_files = sorted(glob.glob(f"{input_path}/**/model_index.json", recursive=True))
    model_index_df = make_model_index_df(model_index_files)

    metadata_files = sorted(glob.glob(f"{input_path}/**/metadata.json", recursive=True))
    metadata_configs = [load_config(metadata_file) for metadata_file in metadata_files]

    df = pd.DataFrame(metadata_configs)

    updated_config_files = sorted(glob.glob(f"{input_path}/**/config_updated.json", recursive=True))
    df["updated_config_path"] = updated_config_files
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