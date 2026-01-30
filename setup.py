"""
Setup script for the stock market prediction project
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="stock-market-prediction",
    version="1.0.0",
    author="Muhammad Farooq",
    author_email="mfarooqshafee333@gmail.com",
    description="A comprehensive ML project for stock market prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Muhammad-Farooq-13/stock-market-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pylint>=2.17.4",
            "black>=23.7.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "stock-market-train=src.models.train_model:main",
            "stock-market-predict=src.models.predict:main",
            "stock-market-api=flask_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
