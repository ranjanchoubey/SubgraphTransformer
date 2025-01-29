from setuptools import setup, find_packages

setup(
    name="graph_transformer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "pyyaml",
    ],
    python_requires=">=3.7",
)