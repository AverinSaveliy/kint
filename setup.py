"""Setup для KINT"""
from setuptools import setup, find_packages

setup(
    name="kint-mega-intelligence",
    version="1.0.0",
    description="KINT - Мегаинтеллект с квантовыми компонентами",
    author="KINT Development",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "sentencepiece>=0.1.99",
        "pennylane>=0.31.0",
        "numpy>=1.24.0",
        "aiohttp>=3.8.0",
        "requests>=2.28.0"
    ],
    entry_points={
        "console_scripts": [
            "kint=main:main"
        ]
    }
)
