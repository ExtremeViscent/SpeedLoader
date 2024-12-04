import os

from setuptools import find_packages, setup


def get_ext_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != ".py":
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)
    return paths


setup(
    name="speedloader",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "wandb",
        "tqdm",
        "matplotlib",
        "seaborn",
        "accelerate",
        "transformers==4.44.0",
        "datasets",
        "deepspeed",
    ],
    author="Yiqi Zhang",
    author_email="limitviscent@gmail.com",
    description="SpeedLoader: An I/O efficient scheme for heterogeneous and distributed LLM operation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
