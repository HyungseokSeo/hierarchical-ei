from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hierarchical-ei",
    version="1.0.0",
    author="Hyungseok Seo",
    author_email="your.email@university.edu",
    description="Hierarchical Emotional Intelligence through JEPA and Active Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HyungseokSeo/hierarchical-ei",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
)