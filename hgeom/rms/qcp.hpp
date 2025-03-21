#include <Eigen/Dense>
#include <iostream>

namespace hgeom {
namespace rms {
namespace qcp {

using std::cout;
using std::endl;
using namespace Eigen;

template <typename F> using RowMatrixX = Matrix<F, Dynamic, Dynamic, RowMajor>;

using F = float;

int qcp_calc_rmsd_maybe_rotation(double *rot, double *A, double *rmsd,
                                 double E0, double len, double minScore) {
  double Sxx, Sxy, Sxz, Syx, Syy, Syz, Szx, Szy, Szz;
  double Szz2, Syy2, Sxx2, Sxy2, Syz2, Sxz2, Syx2, Szy2, Szx2, SyzSzymSyySzz2,
      Sxx2Syy2Szz2Syz2Szy2, Sxy2Sxz2Syx2Szx2, SxzpSzx, SyzpSzy, SxypSyx,
      SyzmSzy, SxzmSzx, SxymSyx, SxxpSyy, SxxmSyy;
  double C[4];
  int i;
  double mxEigenV;
  double oldg = 0.0;
  double b, a, delta, rms, qsqr;
  double q1, q2, q3, q4, normq;
  double a11, a12, a13, a14, a21, a22, a23, a24;
  double a31, a32, a33, a34, a41, a42, a43, a44;
  double a2, x2, y2, z2;
  double xy, az, zx, ay, yz, ax;
  double a3344_4334, a3244_4234, a3243_4233, a3143_4133, a3144_4134, a3142_4132;
  double evecprec = 1e-6;
  double evalprec = 1e-11;

  Sxx = A[0];
  Sxy = A[1];
  Sxz = A[2];
  Syx = A[3];
  Syy = A[4];
  Syz = A[5];
  Szx = A[6];
  Szy = A[7];
  Szz = A[8];

  Sxx2 = Sxx * Sxx;
  Syy2 = Syy * Syy;
  Szz2 = Szz * Szz;

  Sxy2 = Sxy * Sxy;
  Syz2 = Syz * Syz;
  Sxz2 = Sxz * Sxz;

  Syx2 = Syx * Syx;
  Szy2 = Szy * Szy;
  Szx2 = Szx * Szx;

  SyzSzymSyySzz2 = 2.0 * (Syz * Szy - Syy * Szz);
  Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2;

  C[2] = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2);
  C[1] = 8.0 * (Sxx * Syz * Szy + Syy * Szx * Sxz + Szz * Sxy * Syx -
                Sxx * Syy * Szz - Syz * Szx * Sxy - Szy * Syx * Sxz);

  SxzpSzx = Sxz + Szx;
  SyzpSzy = Syz + Szy;
  SxypSyx = Sxy + Syx;
  SyzmSzy = Syz - Szy;
  SxzmSzx = Sxz - Szx;
  SxymSyx = Sxy - Syx;
  SxxpSyy = Sxx + Syy;
  SxxmSyy = Sxx - Syy;
  Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2;

  C[0] = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2 +
         (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) *
             (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2) +
         (-(SxzpSzx) * (SyzmSzy) + (SxymSyx) * (SxxmSyy - Szz)) *
             (-(SxzmSzx) * (SyzpSzy) + (SxymSyx) * (SxxmSyy + Szz)) +
         (-(SxzpSzx) * (SyzpSzy) - (SxypSyx) * (SxxpSyy - Szz)) *
             (-(SxzmSzx) * (SyzmSzy) - (SxypSyx) * (SxxpSyy + Szz)) +
         (+(SxypSyx) * (SyzpSzy) + (SxzpSzx) * (SxxmSyy + Szz)) *
             (-(SxymSyx) * (SyzmSzy) + (SxzpSzx) * (SxxpSyy + Szz)) +
         (+(SxypSyx) * (SyzmSzy) + (SxzmSzx) * (SxxmSyy - Szz)) *
             (-(SxymSyx) * (SyzpSzy) + (SxzmSzx) * (SxxpSyy - Szz));

  /* Newton-Raphson */
  mxEigenV = E0;
  for (i = 0; i < 50; ++i) {
    oldg = mxEigenV;
    x2 = mxEigenV * mxEigenV;
    b = (x2 + C[2]) * mxEigenV;
    a = b + C[1];
    delta = ((a * mxEigenV + C[0]) / (2.0 * x2 * mxEigenV + b + a));
    mxEigenV -= delta;
    /* printf("\n diff[%3d]: %16g %16g %16g", i, mxEigenV - oldg,
     * evalprec*mxEigenV, mxEigenV); */
    if (fabs(mxEigenV - oldg) < fabs(evalprec * mxEigenV))
      break;
  }

  if (i == 50)
    fprintf(stderr, "\nMore than %d iterations needed!\n", i);

  /* the fabs() is to guard against extremely small, but *negative* numbers due
   * to floating point error */
  rms = sqrt(fabs(2.0 * (E0 - mxEigenV) / len));
  (*rmsd) = rms;
  /* printf("\n\n %16g %16g %16g \n", rms, E0, 2.0 * (E0 - mxEigenV)/len); */

  if (minScore > 0 || rot == nullptr)
    if (rms < minScore || rot == nullptr)
      return (-1); // Don't bother with rotation.

  a11 = SxxpSyy + Szz - mxEigenV;
  a12 = SyzmSzy;
  a13 = -SxzmSzx;
  a14 = SxymSyx;
  a21 = SyzmSzy;
  a22 = SxxmSyy - Szz - mxEigenV;
  a23 = SxypSyx;
  a24 = SxzpSzx;
  a31 = a13;
  a32 = a23;
  a33 = Syy - Sxx - Szz - mxEigenV;
  a34 = SyzpSzy;
  a41 = a14;
  a42 = a24;
  a43 = a34;
  a44 = Szz - SxxpSyy - mxEigenV;
  a3344_4334 = a33 * a44 - a43 * a34;
  a3244_4234 = a32 * a44 - a42 * a34;
  a3243_4233 = a32 * a43 - a42 * a33;
  a3143_4133 = a31 * a43 - a41 * a33;
  a3144_4134 = a31 * a44 - a41 * a34;
  a3142_4132 = a31 * a42 - a41 * a32;
  q1 = a22 * a3344_4334 - a23 * a3244_4234 + a24 * a3243_4233;
  q2 = -a21 * a3344_4334 + a23 * a3144_4134 - a24 * a3143_4133;
  q3 = a21 * a3244_4234 - a22 * a3144_4134 + a24 * a3142_4132;
  q4 = -a21 * a3243_4233 + a22 * a3143_4133 - a23 * a3142_4132;

  qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

  /* The following code tries to calculate another column in the adjoint matrix
     when the norm of the current column is too small. Usually this block will
     never be activated.  To be absolutely safe this should be uncommented, but
     it is most likely unnecessary.
  */
  if (qsqr < evecprec) {
    q1 = a12 * a3344_4334 - a13 * a3244_4234 + a14 * a3243_4233;
    q2 = -a11 * a3344_4334 + a13 * a3144_4134 - a14 * a3143_4133;
    q3 = a11 * a3244_4234 - a12 * a3144_4134 + a14 * a3142_4132;
    q4 = -a11 * a3243_4233 + a12 * a3143_4133 - a13 * a3142_4132;
    qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

    if (qsqr < evecprec) {
      double a1324_1423 = a13 * a24 - a14 * a23,
             a1224_1422 = a12 * a24 - a14 * a22;
      double a1223_1322 = a12 * a23 - a13 * a22,
             a1124_1421 = a11 * a24 - a14 * a21;
      double a1123_1321 = a11 * a23 - a13 * a21,
             a1122_1221 = a11 * a22 - a12 * a21;

      q1 = a42 * a1324_1423 - a43 * a1224_1422 + a44 * a1223_1322;
      q2 = -a41 * a1324_1423 + a43 * a1124_1421 - a44 * a1123_1321;
      q3 = a41 * a1224_1422 - a42 * a1124_1421 + a44 * a1122_1221;
      q4 = -a41 * a1223_1322 + a42 * a1123_1321 - a43 * a1122_1221;
      qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

      if (qsqr < evecprec) {
        q1 = a32 * a1324_1423 - a33 * a1224_1422 + a34 * a1223_1322;
        q2 = -a31 * a1324_1423 + a33 * a1124_1421 - a34 * a1123_1321;
        q3 = a31 * a1224_1422 - a32 * a1124_1421 + a34 * a1122_1221;
        q4 = -a31 * a1223_1322 + a32 * a1123_1321 - a33 * a1122_1221;
        qsqr = q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4;

        if (qsqr < evecprec) {
          /* if qsqr is still too small, return the identity matrix. */
          rot[0] = rot[4] = rot[8] = 1.0;
          rot[1] = rot[2] = rot[3] = rot[5] = rot[6] = rot[7] = 0.0;

          return (0);
        }
      }
    }
  }

  normq = sqrt(qsqr);
  q1 /= normq;
  q2 /= normq;
  q3 /= normq;
  q4 /= normq;

  a2 = q1 * q1;
  x2 = q2 * q2;
  y2 = q3 * q3;
  z2 = q4 * q4;

  xy = q2 * q3;
  az = q1 * q4;
  zx = q4 * q2;
  ay = q1 * q3;
  yz = q3 * q4;
  ax = q1 * q2;

  rot[0] = a2 + x2 - y2 - z2;
  rot[1] = 2 * (xy + az);
  rot[2] = 2 * (zx - ay);
  rot[3] = 2 * (xy - az);
  rot[4] = a2 - x2 + y2 - z2;
  rot[5] = 2 * (yz + ax);
  rot[6] = 2 * (zx + ay);
  rot[7] = 2 * (yz - ax);
  rot[8] = a2 - x2 - y2 + z2;

  return (1);
}
template <typename F>
F qcp_rmsd_impl(RowMatrixX<F> xyz1_in, RowMatrixX<F> xyz2_in,
                double *rot = nullptr, double *cen1 = nullptr,
                double *cen2 = nullptr, bool showme = false) {

  if (xyz1_in.rows() != xyz2_in.rows())
    throw std::runtime_error("xyz1 and xyz2 not same size");

  Matrix<F, Dynamic, 3> xyz1 = xyz1_in.block(0, 0, xyz1_in.rows(), 3);
  Matrix<F, Dynamic, 3> xyz2 = xyz2_in.block(0, 0, xyz2_in.rows(), 3);

  auto m1 = xyz1.colwise().mean();
  auto m2 = xyz2.colwise().mean();
  Matrix<F, 1, 3> _cen1(m1);
  Matrix<F, 1, 3> _cen2(m2);
  xyz1.rowwise() -= m1;
  xyz2.rowwise() -= m2;

  // cout << "REF xyz1" << endl;
  // cout << xyz1 << endl;
  // cout << "-----------" << endl;

  auto iprod = xyz1.transpose() * xyz2;

  double E0 = (xyz1.array().square().sum() + xyz2.array().square().sum()) / 2;
  double A[9];
  for (int ii = 0; ii < 3; ++ii)
    for (int jj = 0; jj < 3; ++jj)
      A[3 * ii + jj] = iprod(ii, jj);
  double rmsd;

  if (showme) {
    cout << "REF CEN1 " << _cen1 << endl;
    cout << "REF CEN2 " << _cen2 << endl;
    cout << "REF IPROD" << endl;
    cout << iprod << endl;
    cout << "REF sqnorm1 " << xyz1.array().square().sum() << endl;
    cout << "REF sqnorm2 " << xyz2.array().square().sum() << endl;
    cout << "REF E0 " << E0 << " xyz1.rows() " << xyz1.rows() << endl;
  }

  qcp_calc_rmsd_maybe_rotation(rot, A, &rmsd, E0, xyz1.rows(), -1);

  if (cen1 != nullptr)
    for (int i = 0; i < 3; ++i) {
      cen1[i] = _cen1[i];
    }
  if (cen2 != nullptr)
    for (int i = 0; i < 3; ++i)
      cen2[i] = _cen2[i];

  return rmsd;
}

template <typename F> F qcp_rmsd(RowMatrixX<F> xyz1, RowMatrixX<F> xyz2) {
  double rot[9], c1[3], c2[3];
  return qcp_rmsd_impl(xyz1, xyz2, rot, c1, c2);
}

} // namespace qcp
} // namespace rms
} // namespace hgeom
