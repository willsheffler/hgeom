cmake_minimum_required(VERSION 3.15)
project(hgeom LANGUAGES CXX)

# set(CMAKE_C_COMPILER_LAUNCHER "ccache")
# set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")

# Enable C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)

add_compile_options(-w)

# Find Python and pybind11
find_package(pybind11 REQUIRED)
# find_package(Python COMPONENTS Interpreter Development )
# if(NOT Python_FOUND)
# set(Python_EXECUTABLE "/opt/python/cp313-cp313/bin/python")
# set(Python_INCLUDE_DIR "/opt/python/cp313-cp313/include/python3.13")
# set(Python_LIBRARIES "/opt/python/cp313-cp313/lib/libpython3.13.so")
# endif()

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/hgeom/extern)

pybind11_add_module(_bvh MODULE hgeom/bvh/bvh.cpp)
set_target_properties(_bvh PROPERTIES PREFIX "" OUTPUT_NAME "_bvh" )
target_link_libraries(_bvh PRIVATE pybind11::module)
install(TARGETS _bvh; DESTINATION hgeom)

# pybind11_add_module(_bvh_nd MODULE hgeom/bvh/bvh_nd.cpp)
# set_target_properties(_bvh_nd PROPERTIES PREFIX "" OUTPUT_NAME "_bvh_nd" )
# target_link_libraries(_bvh_nd PRIVATE pybind11::module)
# install(TARGETS _bvh_nd; DESTINATION hgeom)

pybind11_add_module(_cookie_cutter MODULE hgeom/cluster/cookie_cutter.cpp)
set_target_properties(_cookie_cutter PROPERTIES PREFIX "" OUTPUT_NAME "_cookie_cutter" )
target_link_libraries(_cookie_cutter PRIVATE pybind11::module)
install(TARGETS _cookie_cutter; DESTINATION hgeom)

pybind11_add_module(_bcc MODULE hgeom/geom/bcc.cpp)
set_target_properties(_bcc PROPERTIES PREFIX "" OUTPUT_NAME "_bcc" )
target_link_libraries(_bcc PRIVATE pybind11::module)
install(TARGETS _bcc; DESTINATION hgeom)

pybind11_add_module(_expand_xforms MODULE hgeom/geom/expand_xforms.cpp)
set_target_properties(_expand_xforms PROPERTIES PREFIX "" OUTPUT_NAME "_expand_xforms" )
target_link_libraries(_expand_xforms PRIVATE pybind11::module)
install(TARGETS _expand_xforms; DESTINATION hgeom)

pybind11_add_module(_phmap MODULE hgeom/phmap/phmap.cpp)
set_target_properties(_phmap PROPERTIES PREFIX "" OUTPUT_NAME "_phmap" )
target_link_libraries(_phmap PRIVATE pybind11::module)
install(TARGETS _phmap; DESTINATION hgeom)

pybind11_add_module(_qcp MODULE hgeom/rms/qcp.cpp)
set_target_properties(_qcp PROPERTIES PREFIX "" OUTPUT_NAME "_qcp" )
target_link_libraries(_qcp PRIVATE pybind11::module)
install(TARGETS _qcp; DESTINATION hgeom)

pybind11_add_module(_xbin MODULE hgeom/xbin/xbin.cpp)
set_target_properties(_xbin PROPERTIES PREFIX "" OUTPUT_NAME "_xbin" )
target_link_libraries(_xbin PRIVATE pybind11::module)
install(TARGETS _xbin; DESTINATION hgeom)

pybind11_add_module(_xbin_util MODULE hgeom/xbin/xbin_util.cpp)
set_target_properties(_xbin_util PROPERTIES PREFIX "" OUTPUT_NAME "_xbin_util" )
target_link_libraries(_xbin_util PRIVATE pybind11::module)
install(TARGETS _xbin_util; DESTINATION hgeom)

pybind11_add_module(_xform_dist MODULE hgeom/geom/xform_dist.cpp)
set_target_properties(_xform_dist PROPERTIES PREFIX "" OUTPUT_NAME "_xform_dist" )
target_link_libraries(_xform_dist PRIVATE pybind11::module)
install(TARGETS _xform_dist; DESTINATION hgeom)
