from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuravex",
    version="0.1.0",
    author="ELANGKATHIR11",
    description="Neuravex v0.1: Efficient Unified Computer-Vision Architecture with Adaptive Specialization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ELANGKATHIR11/Neuravex",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 (AGPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0"
    ],
    extras_require={
        "coco": ["pycocotools>=2.0.0"]
    }
)
