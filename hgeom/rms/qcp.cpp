/*cppimport
<%


import os
include = os.path.join(os.path.dirname(self.filename),'../..')
cfg['include_dirs'] = [include, f'{include}/hgeom/extern']

cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['qcp.hpp']

cfg['parallel'] = True


setup_pybind11(cfg)
%>
*/

// cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hgeom/rms/qcp.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace hgeom {
namespace rms {
namespace qcp {

template <typename F>
py::tuple qcp_rmsd_align(RowMatrixX<F> xyz1, RowMatrixX<F> xyz2) {
  Matrix<double, 3, 3> R;
  Matrix<double, 3, 1> c1, c2, T;
  double rms;
  {
    py::gil_scoped_release release;
    rms = qcp_rmsd_impl(xyz1, xyz2, R.data(), c1.data(), c2.data());
    T = R * -c1 + c2;
  }
  return py::make_tuple(rms, R, T);
}

template <typename F, typename I>
inline void add_to_center(Matrix<F, 1, 3> &cen,
                          Matrix<F, Dynamic, 3> const &xyz, I offset, I size,
                          int junct = 0) {
  if (junct == 0 || size <= 2 * junct) {
    cen += xyz.block(offset, 0, size, 3).colwise().sum();
  } else {
    cen += xyz.block(offset, 0, junct, 3).colwise().sum();
    cen += xyz.block(offset + size - junct, 0, junct, 3).colwise().sum();
  }
}
template <typename F, typename I>
inline void add_to_sqnorm(F &sqnorm, Matrix<F, Dynamic, 3> const &xyz, I offset,
                          I size, int junct = 0) {
  if (junct == 0 || size <= 2 * junct) {
    sqnorm += xyz.block(offset, 0, size, 3).array().square().sum();
  } else {
    sqnorm += xyz.block(offset, 0, junct, 3).array().square().sum();
    sqnorm +=
        xyz.block(offset + size - junct, 0, junct, 3).array().square().sum();
  }
}

template <typename F, typename I>
inline void add_to_iprod(Matrix<F, 3, 3> &iprod, F &sqnorm,
                         Matrix<F, Dynamic, 3> const &xyz1,
                         Matrix<F, Dynamic, 3> const &xyz2, I offset1,
                         I offset2, I size, Matrix<F, 1, 3> cen1, int junct) {
  if (junct == 0 || size <= 2 * junct) {
    Matrix<F, Dynamic, 3> block1 =
        xyz1.block(offset1, 0, size, 3).rowwise() - cen1;
    Matrix<F, Dynamic, 3> block2 = xyz2.block(offset2, 0, size, 3);
    iprod += block1.transpose() * block2;
    sqnorm += block1.array().square().sum();
  } else {
    {
      Matrix<F, Dynamic, 3> block1 =
          xyz1.block(offset1, 0, junct, 3).rowwise() - cen1;
      Matrix<F, Dynamic, 3> block2 = xyz2.block(offset2, 0, junct, 3);
      iprod += block1.transpose() * block2;
      sqnorm += block1.array().square().sum();
    }
    {
      Matrix<F, Dynamic, 3> block1 =
          xyz1.block(offset1 + size - junct, 0, junct, 3).rowwise() - cen1;
      Matrix<F, Dynamic, 3> block2 =
          xyz2.block(offset2 + size - junct, 0, junct, 3);
      iprod += block1.transpose() * block2;
      sqnorm += block1.array().square().sum();
    }
  }
}
template <typename F, typename I>
py::array_t<F> qcp_rmsd_regions(RowMatrixX<F> xyz1_in, RowMatrixX<F> xyz2_in,
                                Matrix<I, 1, Dynamic> sizes,
                                RowMatrixX<I> offsets, int junct = 0) {
  F *rms = new F[offsets.rows()];
  {
    Matrix<F, Dynamic, 3> xyz1 = xyz1_in.block(0, 0, xyz1_in.rows(), 3);
    Matrix<F, Dynamic, 3> xyz2 = xyz2_in.block(0, 0, xyz2_in.rows(), 3);

    py::gil_scoped_release release;
    if (sizes.rows() != 1 || sizes.cols() != offsets.cols() ||
        sizes.sum() != xyz2.rows())
      throw std::runtime_error("bad sizes or offsets");
    if (junct < 0)
      throw std::runtime_error("junct must be >= 0");

    int nseg = sizes.cols();

    Matrix<I, 1, Dynamic> offsets2(nseg);
    offsets2.fill(0);
    for (int i = 0; i < nseg - 1; ++i)
      offsets2(0, i + 1) = offsets2(0, i) + sizes(0, i);

    int ncrd = 0;
    Matrix<F, 1, 3> cen2(0, 0, 0);
    for (int iseg = 0; iseg < nseg; ++iseg) {
      auto s = sizes(0, iseg);
      ncrd += ((s > (2 * junct)) && junct > 0) ? 2 * junct : s;
      add_to_center(cen2, xyz2, offsets2(0, iseg), sizes(0, iseg), junct);
    }
    cen2 /= ncrd; // xyz2.rows();
    xyz2.rowwise() -= cen2;
    F sqnorm2 = 0; // xyz2.array().square().sum();
    for (int iseg = 0; iseg < nseg; ++iseg) {
      add_to_sqnorm(sqnorm2, xyz2, offsets2(0, iseg), sizes(0, iseg), junct);
    }

    for (int ioff = 0; ioff < offsets.rows(); ++ioff) {
      Matrix<F, 1, 3> cen1(0, 0, 0);
      for (int iseg = 0; iseg < nseg; ++iseg) {
        add_to_center(cen1, xyz1, offsets(ioff, iseg), sizes(0, iseg), junct);
      }
      cen1 /= ncrd; // xyz2.rows();
      // if (junct > 0) {
      // cout << "cen2 " << cen2 << endl;
      // cout << "cen1 " << cen1 << endl;
      // }

      F sqnorm = 0;
      Matrix<F, 3, 3> iprod;
      iprod << 0, 0, 0, 0, 0, 0, 0, 0, 0;
      for (int iseg = 0; iseg < nseg; ++iseg) {
        add_to_iprod(iprod, sqnorm, xyz1, xyz2, offsets(ioff, iseg),
                     offsets2(0, iseg), sizes(0, iseg), cen1, junct);
      }

      double E0 = (sqnorm + sqnorm2) / 2;
      double rmsd;
      double A[9];
      for (int ii = 0; ii < 3; ++ii)
        for (int jj = 0; jj < 3; ++jj)
          A[3 * ii + jj] = iprod(ii, jj);
      qcp_calc_rmsd_maybe_rotation(NULL, A, &rmsd, E0, ncrd, -1);

      rms[ioff] = rmsd;
    }
  }
  py::capsule free_when_done(
      rms, [](void *f) { delete[] reinterpret_cast<F *>(f); });
  return py::array_t<F>({offsets.rows()}, {sizeof(F)}, rms, free_when_done);
}

template <typename F>
py::array_t<double> qcp_rmsd_vec(py::array_t<F> const &a,
                                 RowMatrixX<F> const &b) {
  if (a.ndim() != 3)
    throw std::runtime_error("ndim must be 3");
  if (b.rows() != a.shape()[1])
    throw std::runtime_error("arrays must be same size");
  int M = a.shape()[0];
  int N = a.shape()[1];
  F *ptr = (F *)a.request().ptr;
  F *rms = new F[M];

  for (int i = 0; i < M; ++i) {
    size_t ofst = i * a.strides()[0] / sizeof(F);
    Map<RowMatrixX<F>> xyz2_in(ptr + ofst, N, 3);
    if (b.rows() != xyz2_in.rows())
      throw std::runtime_error("xyz1 and xyz2 not same size");
    Matrix<F, Dynamic, 3> xyz1 = xyz2_in.block(0, 0, xyz2_in.rows(), 3);
    Matrix<F, Dynamic, 3> xyz2 = b.block(0, 0, b.rows(), 3);
    auto m1 = xyz1.colwise().mean();
    auto m2 = xyz2.colwise().mean();
    Matrix<F, 1, 3> _cen1(m1);
    Matrix<F, 1, 3> _cen2(m2);
    xyz1.rowwise() -= m1;
    xyz2.rowwise() -= m2;
    auto iprod = xyz1.transpose() * xyz2;
    double E0 = (xyz1.array().square().sum() + xyz2.array().square().sum()) / 2;
    double A[9];
    for (int ii = 0; ii < 3; ++ii)
      for (int jj = 0; jj < 3; ++jj)
        A[3 * ii + jj] = iprod(ii, jj);
    double *rot = nullptr;
    double rmstmp;
    qcp_calc_rmsd_maybe_rotation(rot, A, &rmstmp, E0, xyz1.rows(), -1);
    rms[i] = rmstmp;
  }
  py::capsule free_when_done(
      rms, [](void *f) { delete[] reinterpret_cast<F *>(f); });
  return py::array_t<F>({M}, {sizeof(F)}, rms, free_when_done);
}

template <typename F>
py::tuple qcp_rmsd_align_vec(py::array_t<F> const &a, RowMatrixX<F> const &b) {
  if (a.ndim() != 3)
    throw std::runtime_error("ndim must be 3");
  if (b.rows() != a.shape()[1])
    throw std::runtime_error("arrays must be same size");
  int M = a.shape()[0];
  int N = a.shape()[1];
  F *ptr = (F *)a.request().ptr;
  F *rms = new F[M];
  F *rot = new F[M * 3 * 3];
  F *trans = new F[M * 3];

  for (int i = 0; i < M; ++i) {
    size_t ofst = i * a.strides()[0] / sizeof(F);
    Map<RowMatrixX<F>> xyz2_in(ptr + ofst, N, 3);
    if (b.rows() != xyz2_in.rows())
      throw std::runtime_error("xyz1 and xyz2 not same size");
    Matrix<F, Dynamic, 3> xyz1 = xyz2_in.block(0, 0, xyz2_in.rows(), 3);
    Matrix<F, Dynamic, 3> xyz2 = b.block(0, 0, b.rows(), 3);
    auto m1 = xyz1.colwise().mean();
    auto m2 = xyz2.colwise().mean();
    Matrix<F, 1, 3> _cen1(m1);
    Matrix<F, 1, 3> _cen2(m2);
    xyz1.rowwise() -= m1;
    xyz2.rowwise() -= m2;
    auto iprod = xyz1.transpose() * xyz2;
    double E0 = (xyz1.array().square().sum() + xyz2.array().square().sum()) / 2;
    double A[9];
    for (int ii = 0; ii < 3; ++ii)
      for (int jj = 0; jj < 3; ++jj)
        A[3 * ii + jj] = iprod(ii, jj);
    double rottmp[9];
    double rmstmp;
    qcp_calc_rmsd_maybe_rotation(&rottmp[0], A, &rmstmp, E0, xyz1.rows(), -1);
    rms[i] = rmstmp;
    for (int j = 0; j < 9; ++j)
      rot[i * 9 + j] = rottmp[j];
    Map<Matrix<F, 3, 3>> Rtmp(rot + i * 9);
    auto Ttmp = Rtmp * -_cen1.transpose() + _cen2.transpose();
    trans[i * 3 + 0] = Ttmp[0];
    trans[i * 3 + 1] = Ttmp[1];
    trans[i * 3 + 2] = Ttmp[2];
  }
  py::capsule freerms(rms, [](void *f) { delete[] reinterpret_cast<F *>(f); });
  auto pyrms = py::array_t<F>({M}, {sizeof(F)}, rms, freerms);
  py::capsule freerot(rot, [](void *f) { delete[] reinterpret_cast<F *>(f); });
  auto pyrot = py::array_t<F>(
      {M, 3, 3}, {3 * 3 * sizeof(F), sizeof(F), 3 * sizeof(F)}, rot, freerot);
  py::capsule freetrans(trans,
                        [](void *f) { delete[] reinterpret_cast<F *>(f); });
  auto pytrans =
      py::array_t<F>({M, 3}, {3 * sizeof(F), sizeof(F)}, trans, freetrans);
  return py::make_tuple(pyrms, pyrot, pytrans);
}

PYBIND11_MODULE(_qcp, m) {
  m.def("qcp_rms_float", &qcp_rmsd<float>, "xyz1"_a, "xyz2"_a);
  m.def("qcp_rms_vec_float", &qcp_rmsd_vec<float>, "xyz1"_a, "xyz2"_a);
  m.def("qcp_rms_align_vec_float", &qcp_rmsd_align_vec<float>, "xyz1"_a,
        "xyz2"_a);
  m.def("qcp_rms_align_float", &qcp_rmsd_align<float>);
  m.def("qcp_rms_regions_f4i4", &qcp_rmsd_regions<float, int32_t>, "xyz1"_a,
        "xyz2"_a, "sizes"_a, "offsets"_a, "junct"_a = 0);
  m.def("qcp_rms_double", &qcp_rmsd<double>);
  m.def("qcp_rms_vec_double", &qcp_rmsd_vec<double>);
  m.def("qcp_rms_align_double", &qcp_rmsd_align<double>);
  m.def("qcp_rms_align_vec_double", &qcp_rmsd_align_vec<double>);
}

} // namespace qcp
} // namespace rms
} // namespace hgeom
