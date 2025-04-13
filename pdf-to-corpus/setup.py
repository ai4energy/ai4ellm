from setuptools import setup, find_packages

setup(
    name="ai4e-refinetext",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "numpy",
        "sentence-transformers",
        "PyPDF2",
        "comtypes",
        "magic-pdf",
        "loguru"
    ],
    author="yan cong",
    description="A pipeline for document OCR, cleaning, semantic deduplication, and conversion to JSONL.",
    python_requires=">=3.8",
)
