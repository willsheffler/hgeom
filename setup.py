from glob import glob
from setuptools import setup, find_packages
from setuptools.command.install import install
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext, intree_extensions

ext_modules = intree_extensions([
    'willutil_cpp/bvh/bvh.cpp',
    'willutil_cpp/bvh/bvh_test.cpp',
    'willutil_cpp/bvh/bvh_nd.cpp',
    'willutil_cpp/geom/miniball.cpp',
    'willutil_cpp/geom/expand_xforms.cpp',
    'willutil_cpp/geom/xform_dist.cpp',
    'willutil_cpp/geom/bcc.cpp',
    'willutil_cpp/geom/primitive_test.cpp',
    'willutil_cpp/util/pybind_types_test.cpp',
    'willutil_cpp/util/dilated_int_test.cpp',
    'willutil_cpp/cluster/cookie_cutter.cpp',
    'willutil_cpp/rms/qcptest.cpp',
    'willutil_cpp/rms/qcp.cpp',
    'willutil_cpp/phmap/phmap.cpp',
    'willutil_cpp/xbin/xbin.cpp',
    'willutil_cpp/xbin/xbin_test.cpp',
    # # 'willutil_cpp/xbin/smear.cpp',
    'willutil_cpp/xbin/xbin_util.cpp',
])
for e in ext_modules:
    e.include_dirs.append('.')
    e.include_dirs.append('willutil_cpp/extern')
    e.extra_compile_args += ['-w']
    # e.extra_compile_args += ['-std=c++17', '-O3', '-march=native', '-fPIC', '-w']

# ParallelCompile("NPY_NUM_BUILD_JOBS", default=8).install()

setup(
    name='willutil_cpp',
    version='0.2',
    url='https://github.com/willsheffler/willutil_cpp',
    author='Will Sheffler',
    author_email='willsheffler@gmail.com',
    description='Wrapped Cpp utils',
    packages=find_packages(include=['willutil_cpp', 'willutil_cpp.*']),
    test_suite='pytest',
    tests_require=['pytest'],
    install_requires=[
        'numpy',
        'icecream',
    ],
    cmdclass={"build_ext": build_ext, 'install': install},
    ext_modules=ext_modules,
    #      package_data={
    #          'willutil_cpp': [
    #              'bvh/bvh.hpp',
    #              'bvh/bvh_algo.hpp',
    #              'geom/bcc.hpp',
    #              'geom/primitive.hpp',
    #              'geom/miniball.hpp',
    #              'util/template_math.hpp',
    #              'util/str.hpp',
    #              'util/global_rng.hpp',
    #              'util/numeric.hpp',
    #              'util/pybind_types.hpp',
    #              'util/Timer.hpp',
    #              'util/dilated_int.hpp',
    #              'util/types.hpp',
    #              'util/assertions.hpp',
    #              'rms/qcp.hpp',
    #              'phmap/phmap.hpp',
    #              'xbin/xbin.hpp',
    #              'xbin/smear.hpp',
    #              'bvh/bvh.cpp',
    #              'bvh/bvh_test.cpp',
    #              'bvh/bvh_nd.cpp',
    #              'geom/miniball.cpp',
    #              'geom/expand_xforms.cpp',
    #              'geom/xform_dist.cpp',
    #              'geom/bcc.cpp',
    #              'geom/primitive_test.cpp',
    #              'util/pybind_types_test.cpp',
    #              'util/dilated_int_test.cpp',
    #              'cluster/cookie_cutter.cpp',
    #              'rms/qcptest.cpp',
    #              'rms/qcp.cpp',
    #              'phmap/phmap.cpp',
    #              'xbin/xbin.cpp',
    #              'xbin/xbin_test.cpp',
    #              'xbin/smear.cpp',
    #              'xbin/xbin_util.cpp',
    #          ]
    #      }
)
