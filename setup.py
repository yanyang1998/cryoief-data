import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cryoief-data",
    version="0.0.1",
    author="Yang Yan",
    author_email="y.yan98@outlook.com",
    description="Cryo-EM data processing tools for deep learning (e.g., cryo-IEF)",
    long_description=long_description,
    long_description_content_type="README.md",
    url="https://github.com/yanyang1998/cryoief-data",
    project_urls={
        "Homepage": "https://github.com/yanyang1998/cryoief-data",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",

    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    intall_requires=['accelerate>=0.29.0',
                     'annoy>=1.17.0',
                     'asposestorage>=1.0.0',
                     'cryosparc_tools>=4.4.0',
                     'lmdb>=1.6.0',
                     'matplotlib>=3.8.0',
                     'mrcfile>=1.5.0',
                     'munkres>=1.1.0',
                     'natsort>=8.4.0',
                     'numba>=0.59.0',
                     'numpy>=2.3.0',
                     'pandas>=2.3.0',
                     'Pillow>=11.2.0',
                     'pyFFTW>=0.13.0',
                     'scikit_learn>=1.4.0',
                     'scipy>=1.15.0',
                     'seaborn>=0.13.0',
                     'setuptools>=69.5.0',
                     'torch>=2.3.0',
                     'torchvision>=0.18.0',
                     'tqdm>=4.66.0']
)
