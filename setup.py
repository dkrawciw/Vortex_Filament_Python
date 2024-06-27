from setuptools import find_packages, setup


setup(
    name='filamentlib',
    version='0.1.1',
    description='Simulate the evolution of any given vortex filament over time',
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Daniel Krawciw',
    author_email='dkrawciw@mines.edu',
    license='MIT',
    python_requires='>=3.10',
    install_requires=['numpy >= 1.26','scipy >= 1.13.0']
)