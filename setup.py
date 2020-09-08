from setuptools import setup
from setuptools.extern import packaging
import ast

with open('onfire/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = ast.parse(line).body[0].value.s
            version = str(packaging.version.Version(version))
            break

with open('README.md') as f:
    long_description = f.read()

setup(
    name='pytorch-onfire',
    version=version,
    author='Jose Fernandez Portal, Rafael Carrascosa',
    author_email='jose.fp@gmail.com',
    description='PyTorch meets Sklearn Pipelines.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/joshfp/pytorch-onfire',
    license='MIT',
    packages=[
        'onfire',
        'onfire.colab',
    ],
    install_requires=[
        'torch',
        'scikit-learn',
        'Unidecode',
        'lmdb',
        'msgpack',
        'fastprogress',
        'matplotlib',
    ],
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
