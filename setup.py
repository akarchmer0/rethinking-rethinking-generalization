"""Setup script for the rethinking-generalization package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Comprehensive experimental framework challenging Zhang et al. (2017) on deep learning generalization."

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
        "jupyter>=1.0.0",
        "pytest>=7.4.0",
    ]

setup(
    name="rethinking-generalization",
    version="0.1.0",
    author="Research Team",
    author_email="your.email@example.com",
    description="Challenging Zhang et al. (2017) on deep learning generalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rethinking-generalization-rebuttal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-baseline=experiments.baseline_replication:main",
            "run-smoothness=experiments.smoothness_analysis:main",
            "run-two-stage=experiments.two_stage_learning:main",
            "run-frequency=experiments.frequency_analysis:main",
            "run-corruption=experiments.complexity_measures:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

