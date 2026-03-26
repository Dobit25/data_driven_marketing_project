"""
setup.py — Time Series Forecasting Project
Enables `pip install -e .` for editable installs.
"""

from setuptools import find_packages, setup

setup(
    name="time_series_project",
    version="1.0.0",
    description="Forecasting concurrent video game players using time series models",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/<your-org>/time_series_project",
    packages=find_packages(),
    python_requires=">=3.9",
)
