"""
Setup file for package `PyVtools`.
"""
from setuptools import setup
import pathlib

PACKAGENAME = 'PyVtools'

# the directory where this setup.py resides

HERE = pathlib.Path(__file__).absolute().parent

# function to parse the version from

def read_version():
    with (HERE / PACKAGENAME / '__init__.py').open() as fid:
        for line in fid:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")

if __name__ == "__main__":

    setup(
        name=PACKAGENAME,
        description='Functions to handle FITS files.',
        version=read_version(),
        long_description=(HERE / "README.md").read_text(),
        long_description_content_type='text/markdown',
        url='https://github.com/simoncasassus/' + PACKAGENAME,
        author='Simon Casassus',
        author_email='simoncasassus@gmail.com',
        license='GPLv3',
        packages=[PACKAGENAME],
        install_requires=[ # Modify for current packages
            'numpy',
            'astropy',
            'matplotlib'],
        python_requires='>=3.6',
        entry_points={
            'console_scripts': [
                'myfuncshello = myfuncs.script:main',
            ],
        }
    )