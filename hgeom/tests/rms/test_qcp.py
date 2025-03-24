import hgeom as wu
from hgeom import qcp_rms_double, qcp_rms_vec_double, qcp_rms_align_double, qcp_rms_regions_f4i4
from hgeom import qcp_rms_align_vec_double, qcp_rms_align_vec_float, qcp_rms_align_float, qcp_rms_vec_float
import numpy as np

def main():
   test_qcp_vec()
   # test_qcp_align(niter=10)
   # assert 0
   # test_qcp_align()
   test_qcp_align_vec()
   # assert 0

   test_qcp_regions_junct_simple()

   test_qcp_regions()
   test_qcp_regions_junct()

   test_qcp_regions_simple_1seg()
   test_qcp_regions_simple_2seg()
   test_qcp_regions_simple_Nseg()

   test_qcp(niter=10)
   test_qcp_align(niter=10)

   perftest_qcp_regions()

   print('test_qcp PASS', flush=True)

def test_qcp_vec(npts=(1000, 10)):
   pts1 = wu.homog.hrandpoint(npts)[:, :, :3].copy()
   pts2 = wu.homog.hrandpoint(npts[1])[:, :3].copy()
   with wu.Timer():
      rms = qcp_rms_vec_double(pts1, pts2)
   with wu.Timer():
      rms2 = [qcp_rms_double(p1, pts2) for p1 in pts1]
   pts1 = pts1.astype(np.float32)
   pts2 = pts2.astype(np.float32)
   with wu.Timer():
      rms3 = qcp_rms_vec_float(pts1, pts2)
   assert np.allclose(rms, rms2)
   assert np.allclose(rms, rms3)

def test_qcp_align_vec(npts=(1000, 10)):
   pts1 = wu.homog.hrandpoint(npts)[:, :, :3].copy()
   pts2 = wu.homog.hrandpoint(npts[1])[:, :3].copy()
   with wu.Timer():
      rms2, R2, T2 = zip(*[qcp_rms_align_double(p1, pts2) for p1 in pts1])
   rms, R, T = np.stack(rms2), np.stack(R2), np.stack(T2)
   with wu.Timer():
      rms2, R2, T2 = qcp_rms_align_vec_double(pts1, pts2)
   assert np.allclose(rms, rms2)
   assert np.allclose(R, R2)
   assert np.allclose(T, T2)
   pts1 = pts1.astype(np.float32)
   pts2 = pts2.astype(np.float32)
   with wu.Timer():
      rms3, R3, T3 = qcp_rms_align_vec_float(pts1, pts2)

   assert np.allclose(rms, rms3, atol=1e-4)
   assert np.allclose(R, R3, atol=1e-4)
   assert np.allclose(T, T3, atol=1e-4)

def test_qcp_regions_simple_1seg():
   N = 100
   pts1 = wu.homog.hrandpoint(N + 10).astype(np.float32)
   pts2 = wu.homog.hrandpoint(N).astype(np.float32)
   # pts1[:100] = pts1[:100] * 0.01 + pts2 * 0.99
   offsets = np.arange(10).reshape(10, 1)
   # offsets = np.tile(offsets, (170_000, 1))
   # print(offsets.shape)
   # with wu.Timer():
   rms = qcp_rms_regions_f4i4(pts1, pts2, [N], offsets)
   # print(rms)
   for i in range(10):
      rmsref = qcp_rms_double(pts1[i:N + i], pts2)
      assert rms.shape == (len(offsets), )
      assert np.allclose(rms[i], rmsref, atol=1e-4)

def test_qcp_regions_simple_2seg():
   N = 40
   pts1 = wu.homog.hrandpoint(N).astype(np.float32)
   pts2 = wu.homog.hrandpoint(N).astype(np.float32)
   rmsref = qcp_rms_double(pts1, pts2)
   # pts1[:, :3] -= pts1[:, :3].mean(axis=0).reshape(1, 3)
   # pts2[:, :3] -= pts2[:, :3].mean(axis=0).reshape(1, 3)
   for i in range(5, 35):
      sizes = [i, N - i]
      offsets = np.array([[0, i]], dtype='i4')
      rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets)
      assert rms.shape == (len(offsets), )
      assert np.allclose(rms, rmsref, atol=1e-4)

def _random_int_partition(n, r):
   import random
   p = list()
   while sum(p) < n:
      p.append(random.randint(0, r))
   p[-1] = n - sum(p[:-1])
   return p

def _random_offsets(n, l, sizes):
   return np.stack([np.random.randint(0, l - s + 1, n) for s in sizes], axis=-1)

def test_qcp_regions_simple_Nseg():
   N = 40
   pts1 = wu.homog.hrandpoint(N).astype(np.float32)
   pts2 = wu.homog.hrandpoint(N).astype(np.float32)
   rmsref = qcp_rms_double(pts1, pts2)
   # pts1[:, :3] -= pts1[:, :3].mean(axis=0).reshape(1, 3)
   # pts2[:, :3] -= pts2[:, :3].mean(axis=0).reshape(1, 3)
   for i in range(100):
      sizes = _random_int_partition(N, 10)
      offsets = np.cumsum([0] + sizes[:-1]).reshape(1, len(sizes))
      rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets)
      assert rms.shape == (len(offsets), )
      assert np.allclose(rms, rmsref, atol=1e-4)

def compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=0):
   rms = np.empty(len(offsets))

   offsets2 = np.cumsum([0] + list(sizes[:-1]))
   crd2 = list()
   for s, o in zip(sizes, offsets2):
      if junct == 0 or 2 * junct >= s:
         crd2.append(pts2[o:o + s, :3])
      else:
         crd2.append(pts2[o:o + junct, :3])
         crd2.append(pts2[o + s - junct:o + s, :3])
   crd2 = np.concatenate(crd2)
   for i, ofst in enumerate(offsets):
      crd1 = list()
      for s, o in zip(sizes, ofst):
         if junct == 0 or 2 * junct >= s:
            crd1.append(pts1[o:o + s, :3])
         else:
            crd1.append(pts1[o:o + junct, :3])
            crd1.append(pts1[o + s - junct:o + s, :3])
      crd1 = np.concatenate(crd1)

      rms[i] = qcp_rms_double(crd1, crd2)
   return rms

def perftest_qcp_regions():
   t = wu.Timer()
   ncalc = 0
   for i in range(30):
      pts1 = wu.homog.hrandpoint(200).astype(np.float32)
      pts2 = wu.homog.hrandpoint(50).astype(np.float32)
      sizes = _random_int_partition(len(pts2), len(pts2) - 5)
      offsets = _random_offsets(30_000, len(pts1), sizes)
      t.checkpoint('setup')
      rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=0)
      ncalc += len(rms)
      t.checkpoint('qcp_rms_regions_f4i4')
   t.report()
   rmspersec = ncalc / sum(t.checkpoints['qcp_rms_regions_f4i4'])
   print(f'rms ncalc: {ncalc:,}, rate: {rmspersec:7.3}', flush=True)

def helper_test_qcp_regions(noffset=1, junct=0, npts1=100, npts2=50):
   pts1 = wu.homog.hrandpoint(npts1).astype(np.float32)
   pts2 = wu.homog.hrandpoint(npts2).astype(np.float32)
   sizes = _random_int_partition(npts2, npts2 - 5)
   offsets = _random_offsets(noffset, len(pts1), sizes)
   rmsref = compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=junct)
   rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=junct)
   assert np.allclose(rms, rmsref)

def test_qcp_regions():
   for i in range(1000):
      helper_test_qcp_regions(noffset=10, junct=0)

def test_qcp_regions_junct_simple():
   pts1 = wu.homog.hrandpoint(9).astype(np.float32)[:, :3]
   pts2 = wu.homog.hrandpoint(9).astype(np.float32)[:, :3]

   # pts1 -= pts1[:4].mean(axis=0).reshape(-1, 3)
   # pts2 -= pts2[:4].mean(axis=0).reshape(-1, 3)

   sizes = [9]
   offsets = [[0]]
   rmsref = compute_rms_offsets_brute(pts1, pts2, sizes, offsets, junct=4)
   rms = qcp_rms_regions_f4i4(pts1, pts2, sizes, offsets, junct=4)
   # print(rmsref)
   # print(rms)
   assert np.allclose(rms, rmsref)

def test_qcp_regions_junct():
   for j in range(2, 10):
      for i in range(10):
         helper_test_qcp_regions(noffset=10, junct=j)

def test_qcp_align(niter=20, npts=50):
   for i in range(niter):
      pts1 = wu.homog.hrandpoint(npts)
      pts2 = wu.homog.hrandpoint(npts)
      pts1copy, pts2copy = pts1.copy(), pts2.copy()
      rms, fit, x = wu.homog.hrmsfit(pts1, pts2)
      rms2, R, T = qcp_rms_align_double(pts1, pts2)
      assert np.allclose(rms, rms2)
      assert np.allclose(x[:3, :3], R)
      assert np.allclose(x[:3, 3], T)
      assert np.allclose(pts1, pts1copy)
      assert np.allclose(pts2, pts2copy)
      rms2, R, T = qcp_rms_align_float(pts1, pts2)
      assert np.allclose(rms, rms2, atol=1e-4)
      assert np.allclose(x[:3, :3], R, atol=1e-4)
      assert np.allclose(x[:3, 3], T, atol=1e-4)

def test_qcp(niter=100, npts=50):
   for i in range(niter):
      pts1 = wu.homog.hrandpoint(npts)
      pts2 = wu.homog.hrandpoint(npts)
      rms, fit, x = wu.homog.hrmsfit(pts1, pts2)
      rms2 = qcp_rms_double(pts1, pts2)
      assert np.allclose(rms, rms2)

if __name__ == '__main__':
   main()
