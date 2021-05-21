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
        "tensorflow==2.4.1",
        "numpy<=1.20.2",
        "pandas==1.2.3",
        "Keras==2.4.3",
        "cython==0.29.23",
        "tulipy==0.4.0",
        "yfinance==0.1.54",
        "scikit-learn==0.24.2",
    ],
)
