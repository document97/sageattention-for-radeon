from setuptools import find_packages, setup


setup(
    name="sageattention",
    version="2.2.0+hip57core",
    author="SageAttention team; HIP57 core adaptation",
    license="Apache 2.0 License",
    description="Minimal SageAttention2 Triton core for AMD HIP57/ZLUDA ComfyUI inference.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
)
