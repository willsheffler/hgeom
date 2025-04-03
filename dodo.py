import sysconfig
import os


def find_pybind():
    pybind = sysconfig.get_paths()['purelib'] + '/pybind11'
    pybind = f'-Dpybind11_DIR={pybind}'
    print(pybind)
    return pybind


def task_cmake_build():
    """Configure and build the C++ extension via CMake"""
    build_dir = '_build'
    cmake = f'cmake {find_pybind()} -B {build_dir} -DCMAKE_BUILD_TYPE=Release -GNinja'
    print(cmake)
    bld = f'cmake --build {build_dir} --config Release'
    return {
        'actions': ['mkdir -p _build', cmake, bld],
        'targets': [os.path.join(build_dir, 'libyourmodule.so')],
        'clean': True,
        'verbosity': 2,
    }


def task_import_check():
    """Try to import the compiled module to verify it's working"""

    def import_test():
        try:
            from hgeom import BVH as BVH
        except Exception as e:
            print('❌ Import failed:', e)
            raise

    return {
        'actions': [import_test],
        'task_dep': ['cmake_build'],
    }


def task_test():
    """Run tests using pytest"""
    return {
        'actions': ['pytest hgeom/tests'],
        'task_dep': ['import_check'],
    }
