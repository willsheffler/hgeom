from setuptools import setup, find_packages

setup(name='willutil_cpp', version='0.1', url='https://github.com/willsheffler/willutil_cpp', author='Will Sheffler', author_email='willsheffler@gmail.com', description='Wrapped Cpp utils', packages=find_packages(include=['willutil_cpp', 'willutil_cpp.*']), install_requires=[
   'pytest',
   'numpy',
   'cppimport',
],
)
