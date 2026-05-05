from setuptools import setup, find_packages

setup(
    name="MolMap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scipy>=1.5.0",
        "rdkit>=2020.09.1",
    ],
    python_requires=">=3.7",
    author="Your Name",
    description="Molecular mapping and Bayesian optimization for molecular design",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
)