from setuptools import setup, find_packages
import numpy as np


def readme():
    with open('README.rst') as f:
        return f.read()


EXTRAS_REQUIRE = {'tests': ['pytest', 'pytest-cov'],
                  'docs': ['sphinx', 'sphinx-gallery',
                           'sphinx_rtd_theme', 'numpydoc',
                           'matplotlib', 'download']
                  }

if __name__ == "__main__":
    setup(name="groupmne",
          packages=find_packages(),
          include_dirs=[np.get_include()],
          extras_require=EXTRAS_REQUIRE,
          version='0.0.1dev',
          description='Group MNE source localization',
          long_description=readme(),
          classifiers=[
              'Programming Language :: Python :: 3.6',
              'Development Status :: 4 - Beta',
              'Intended Audience :: Developers',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Telecommunications Industry',
              'Natural Language :: English',
          ],
          keywords='MNE MEG EEG source localization multi-task regression',
          url='https://github.com/hichamjanati/groupmne',
          author='Hicham Janati',
          author_email='hicham.janati100@gmail.com',
          license='MIT',
          )
