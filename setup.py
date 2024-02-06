from setuptools import setup, find_packages

setup(
    name="pnet",
    version="0.1.0",
    description="Biologically informed deep neural network model for fast and accurate biological predictions with model built-in interpretability by design",
    author="Marc Glettig",
    author_email="marc.glettig@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List your project's dependencies here
        # e.g., "numpy", "pandas>=1.0"
    ],
)

