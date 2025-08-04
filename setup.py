"""
Setup script for Hierarchical Emotional Intelligence package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hierarchical-ei",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Hierarchical Emotional Intelligence: A Unified JEPA-Active Inference Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hierarchical-ei",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/hierarchical-ei/issues",
        "Documentation": "https://hierarchical-ei.readthedocs.io",
        "Paper": "https://arxiv.org/abs/xxxx.xxxxx",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "demo": [
            "opencv-python>=4.5.0",
            "sounddevice>=0.4.0",
            "pyaudio>=0.2.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "hierarchical-ei-train=hierarchical_ei.train:main",
            "hierarchical-ei-evaluate=hierarchical_ei.evaluate:main",
            "hierarchical-ei-demo=hierarchical_ei.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "hierarchical_ei": ["configs/*.yaml"],
    },
)
