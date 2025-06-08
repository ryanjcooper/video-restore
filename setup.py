# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#") and not line.startswith("--")]

setup(
    name="video-restore",
    version="1.0.0",
    author="Ryan Cooper",
    author_email="your.email@example.com",
    description="AI-powered video upscaling with multi-GPU support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ryanjcooper/video-restore",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video :: Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "video-restore=video_upscaler:main",
        ],
    },
    include_package_data=True,
)