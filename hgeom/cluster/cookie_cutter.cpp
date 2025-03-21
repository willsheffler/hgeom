/*/*cppimport
<%


import os
include = os.path.join(os.path.dirname(self.filename),'../..')
cfg['include_dirs'] = [include, f'{include}/hgeom/extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../util/types.hpp']

cfg['parallel'] = True


setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

namespace hgeom {
/**
\namespace hgeom::cluster
\brief namespace for clustering stuff
*/
namespace cluster {
/**
\namespace hgeom::cluster::cookie_cutter
\brief namespace for cookie_cutter style clustering
*/
namespace cookie_cutter {

using namespace Eigen;

template <typename F> using Vx = Matrix<F, 1, Dynamic>;
template <typename F> using RowMatrixX = Matrix<F, Dynamic, Dynamic, RowMajor>;

template <typename F>
py::tuple cookie_cutter(Ref<RowMatrixX<F>> pts, F thresh) {
  Vx<int> out;
  std::vector<int> clustid;
  {
    py::gil_scoped_release release;
    using Keeper = std::pair<Vx<F>, int>;
    std::vector<Keeper> keep;

    for (int i = 0; i < pts.rows(); ++i) {
      int seenit = -1;
      for (auto &keeper : keep) {
        Vx<F> delta = pts.row(i) - keeper.first;
        F d2 = delta.squaredNorm();
        if (d2 <= thresh * thresh) {
          seenit = keeper.second;
          break;
        }
      }
      if (seenit == -1) {
        keep.push_back(Keeper(pts.row(i), i));
        clustid.push_back(i);
      }

      else
        clustid.push_back(seenit);
    }
    out.resize(keep.size());
    for (int i = 0; i < keep.size(); ++i) {
      out[i] = keep[i].second;
    }
  }
  return py::make_tuple(out, clustid);
}

PYBIND11_MODULE(_cookie_cutter, m) {
  m.def("cookie_cutter", &cookie_cutter<double>);
  m.def("cookie_cutter", &cookie_cutter<float>);
}

} // namespace cookie_cutter
} // namespace cluster
} // namespace hgeom
