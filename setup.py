"""
setup.py — Customer Lifetime Value Prediction (Dunnhumby)
Enables `pip install -e .` for editable installs.
"""

from setuptools import find_packages, setup

setup(
    name="clv_dunnhumby",
    version="1.0.0",
    description="Customer Lifetime Value prediction using Dunnhumby retail transaction data",
    author="DSEB65A Group 6",
    author_email="truonghoangtung25@gmail.com",
    url="https://github.com/DSEB65A-Group6/data_driven_marketing_project",
    packages=find_packages(),
    python_requires=">=3.9",
)
