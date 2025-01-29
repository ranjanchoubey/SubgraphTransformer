from setuptools import setup, find_packages

setup(
    name="graph_transformer",
    version="0.1.0",
    description="Graph Transformer Implementation",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.1",
        "wandb>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.910",
        ]
    }
)
