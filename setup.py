from setuptools import setup, find_packages

setup(
    name="xrl-simulator",
    version="0.1.0",
    description="X-ray lithography process simulator for education and research",
    author="Abhineet Agarwal",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    extras_require={
        "yaml": ["pyyaml>=6.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
