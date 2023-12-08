# Create a Dataset of Configs for `diffusers` Model Components

## Setup Dependencies

```shell
pip install -r requirements.txt
```

## Download Configs

This script will download the configs for a `diffusers` model component, e.g. UNet and save them to a `configs`. It will also load in the config into a UNet model and save the config in the latest format as `config_updated.json`

```shell
python fetch_configs.py
```

Should the fetching process be interrupted, the script also creates a cache of configs already downloaded so that it doesn't have to download them again.


```shell
python fetch_configs.py --cache cache.json
```

Any errors will be written to a `fetch_configs.log` file.

## Build Dataset

Once the configs have been downloaded, we can build a dataset of configs for a model component. This script will create a csv file containing metadata for each config. The csv file contains the following columns:

```shell
'org_id'
'model_name'
'downloads'
'likes'
'class_name' # UNet model class
'config_hash' # hash of the config string in the latest diffusers format
'updated_config_path' # path to the config json in the latest diffusers format
'pipeline' # pipeline that uses this class
```

Some example rows.
```
    org_id                model_name  ...                                updated_config_path                 pipeline
0     5w4n  deliberate-v2-inpainting  ...  configs/5w4n/deliberate-v2-inpainting/unet/con...  StableDiffusionPipeline
1  CompVis     stable-diffusion-v1-4  ...  configs/CompVis/stable-diffusion-v1-4/unet/con...  StableDiffusionPipeline
2  EK12317           Ekmix-Diffusion  ...  configs/EK12317/Ekmix-Diffusion/unet/config_up...  StableDiffusionPipeline
3    Lykon               DreamShaper  ...  configs/Lykon/DreamShaper/unet/config_updated....  StableDiffusionPipeline
4    Meina              MeinaMix_V10  ...  configs/Meina/MeinaMix_V10/unet/config_updated...  StableDiffusionPipeline
```

[Link to example dataset](https://huggingface.co/datasets/diffusers/configs-dataset/tree/main)

This `metatdata.csv` file and all the downloaded configs and converted configs can up uploaded to a dataset repo by passing in a repo id.

```shell
python build_dataset.py --input_path <path to dowloaded configs folder> --repo_id <repo_id>
```

If no repo id is passed in, the script will just create the `metadata.csv` file.
