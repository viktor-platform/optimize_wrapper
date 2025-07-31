from setuptools import setup, find_packages

setup(
    name="optimize_wrapper",
    version="0.1.0",
    description="A wrapper for genetic optimization with Viktor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "munch>=4.0.0",
        "pygad>=3.5.0",
        "viktor>=14.24.0",
        "matplotlib",
        "numpy",
    ],
    python_requires=">=3.8",
) 