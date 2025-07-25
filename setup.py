
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'cryodata'
DESCRIPTION = 'Cryo-EM data processing tools for deep learning (e.g., cryo-IEF).'
URL = 'https://github.com/yanyang1998/cryoief-data'
EMAIL = 'y.yan98@outlook.com'
AUTHOR = 'Yang Yan'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = '0.1.1b01'

# What packages are required for this module to be executed?
REQUIRED = ['accelerate>=1.0.0',
                     'annoy>=1.17.0',
                     'asposestorage>=1.0.0',
                     'cryosparc_tools>=4.4.0',
                     'lmdb>=1.6.0',
                     'matplotlib>=3.8.0',
                     'mrcfile>=1.5.0',
                     'munkres>=1.1.0',
                     'natsort>=8.4.0',
                     'numba>=0.59.0',
                     'numpy>=1.1.7, <3.0.0',
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

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)