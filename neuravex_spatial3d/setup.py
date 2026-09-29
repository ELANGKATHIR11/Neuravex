from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuravex-spatial3d",
    version="0.1.0",
    author="ELANGKATHIR11",
    description="High-performance Native 3D Spatial Perception, LWH Bounding, Camera/LiDAR Sensor Fusion, DEM, and Volumetric Counting SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ELANGKATHIR11/Neuravex",
    packages=["neuravex_spatial3d"],
    package_dir={"neuravex_spatial3d": "."},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU Affero General Public License v3 (AGPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.22.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scipy>=1.9.0",
    ],
    extras_require={
        "cuda": ["torch>=2.0.0"],
        "dev": ["pytest>=7.0.0", "ruff>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "neuravex-spatial3d=neuravex_spatial3d.cli:main",
        ],
    }
)
