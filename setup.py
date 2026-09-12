from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yolo27",
    version="0.6.0",
    author="ELANGKATHIR11",
    description="YOLO27 v0.6: Unified End-to-End Multi-Task Vision Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ELANGKATHIR11/YOLO27",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
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
