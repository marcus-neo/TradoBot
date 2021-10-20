import setuptools

with open("longdescription.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tradobot",
    version="0.1.0",
    author="Marcus Neo",
    author_email="marcus.neo418@gmail.com",
    description="Simple Trading Bot Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://google.com",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    install_requires=[
        "sklearn",
        "keras",
        "tensorflow==2.5.1",
        "numpy",
        "Pandas",
        "yfinance",
        "tulipy",
        "matplotlib",
        "toolz",
        "seaborn",
        "Pillow",
        "pylint",
        "flake8",
        "pydocstyle",
        "pytest",
    ],
)
