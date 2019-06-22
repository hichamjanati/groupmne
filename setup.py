from setuptools import setup, find_packages
import numpy


def readme():
    with open('README.md') as f:
        return f.read()


INSTALL_REQUIRES = ['numpy', 'joblib', 'numba']

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib',
        'download'

    ]
}

if __name__ == "__main__":
    setup(name="groupmne",
          packages=find_packages(),
          version="0.0.1dev",
          include_dirs=[numpy.get_include()],
          install_requires=INSTALL_REQUIRES,
          extras_require=EXTRAS_REQUIRE,
          )
