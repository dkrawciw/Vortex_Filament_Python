from setuptools import find_packages, setup

with open('filamentlib/README.md','r') as f:
    long_description = f.read()

setup(
    name='filamentlib',
    version='0.0.10',
    description='Simulate the evolution of any given vortex filament over time',
    package_dir={'':'filamentlib'},
    packages=find_packages(where='filamentlib'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Daniel Krawciw',
    author_email='dkrawciw@mines.edu',
    license='MIT',
    python_requires='>=3.10',
    install_requires=['numpy >= 1.26','scipy >= 1.13.0']
)