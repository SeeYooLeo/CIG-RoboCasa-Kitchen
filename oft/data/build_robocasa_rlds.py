import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from robocasa_no_noops_dataset_builder import RobocasaNoNoops
import tensorflow_datasets as tfds


def main():
    data_dir = os.environ.get("DATA_DIR", "datasets/rlds")
    os.makedirs(data_dir, exist_ok=True)
    b = RobocasaNoNoops(data_dir=data_dir)
    dl_cfg = tfds.download.DownloadConfig()
    b.download_and_prepare(download_config=dl_cfg)
    print("DONE", b.data_dir)


if __name__ == "__main__":
    main()
