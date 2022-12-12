from setuptools import setup, find_packages

setup(
    name='loftr',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "opencv-python==4.2.0.34",
        "albumentations==0.5.1",
        "ray>=1.0.1",
        "einops==0.3.0",
        "loguru==0.5.3",
        "yacs>=0.1.8",
        "tqdm",
        "pyyaml==5.4.1",
        "timm",
        "h5py==3.1.0",
        "pytorch-lightning==1.3.5",
        "torchmetrics==0.6.0",
        "joblib>=1.0.1",
        "kornia"
    ],
)
