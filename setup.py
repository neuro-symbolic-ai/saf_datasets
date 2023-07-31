from distutils.core import setup

setup(
    name='saf_datasets',
    version='0.1.14',
    packages=['saf_datasets', 'saf_datasets.annotators', 'saf_datasets.data_access'],
    url='',
    license='GNU General Public License v3.0',
    author='Danilo S. Carvalho',
    author_email='danilo.carvalho@manchester.ac.uk',
    description='Simple Annotation Framework - Data set loading facilities',
    install_requires=[
        'saf @ git+https://github.com/dscarvalho/saf.git',
        'spacy',
        'gdown',
        'tqdm',
        'torch',
        'jsonlines',
        'transformers'
    ]
)
