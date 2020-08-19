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

setup(name='pytorch-onfire',
      version=version,
      author='Jose Fernandez Portal',
      author_email='jose.fp@gmail.com',
      description='PyTorch helper library',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/joshfp/onfire',
      license='MIT',
      packages=['onfire'],
      install_requires=['torch', 'scikit-learn', 'Unidecode'],
      python_requires='>=3.6',
)
