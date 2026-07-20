from setuptools import setup, find_packages

setup(
    name="nlp_pipeline",
    version="0.1.0",
    description="Modular NLP pipeline for Privacy-Preserving Federated Hierarchical Multi-Label Clinical Text Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
    ],
)
