import os
from distutils.core import setup

PKG_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements() -> list:
    """Load requirements from file, parse them as a Python list"""

    with open(os.path.join(PKG_ROOT, "requirements.txt"), encoding="utf-8") as f:
        all_reqs = f.read().split("\n")
    install_requires = [str(x).strip() for x in all_reqs]

    return install_requires


setup(
    name='saf_datasets',
    version='0.4.4',
    packages=['saf_datasets', 'saf_datasets.annotators', 'saf_datasets.data_access', 'saf_datasets.wrappers'],
    url='',
    license='GNU General Public License v3.0',
    author='Danilo S. Carvalho',
    author_email='danilo.carvalho@manchester.ac.uk',
    description='Simple Annotation Framework - Data set loading facilities',
    install_requires=load_requirements()
)
