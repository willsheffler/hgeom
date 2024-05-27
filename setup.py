from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        from willutil_cpp import bvh cluster geom phmap rms util xbin

setup(
    name='willutil_cpp',
    version='0.1',
    url='https://github.com/willsheffler/willutil_cpp',
    author='Will Sheffler',
    author_email='willsheffler@gmail.com',
    description='Wrapped Cpp utils',
    packages=find_packages(include=['willutil_cpp', 'willutil_cpp.*']),
    install_requires=[
        'pytest',
        'numpy',
        'cppimport',
    ],
   cmdclass={
        'install': PostInstallCommand,
    },
)
