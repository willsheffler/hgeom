from time import perf_counter
import numpy as np

import pytest

pytest.importorskip('hgeom.bvh_nd')
from hgeom.bvh_nd import *


def test_bvh_isect6():
    nhit, ntot = 0, 0
    for iter in range(1):
        pts = np.random.randn(1_000, 7) + [2, 2, 2, 2, 2, 2, 2]
        bvh = create_bvh7d(pts)
        assert np.allclose(bvh.com(), np.mean(pts, axis=0))
        pts2 = np.random.randn(1_000, 7)
        bvh2 = create_bvh7d(pts2)
        for mindis in np.arange(0.1, 2, 0.1):
            isect1 = bvh_isect7d(bvh, bvh2, mindis)
            isect2 = bvh_isect7d_naive(bvh, bvh2, mindis)
            assert isect1 == isect2
            nhit += isect1
            ntot += 1
    print('frac', nhit / ntot)


def test_bvh_isect7():
    """Run performance tests for bvh_isect7d function.

    This function measures the performance of two intersection functions:
    bvh_isect7d and bvh_isect7d_naive. It randomly generates points and BVH
    structures, performs intersection checks, and compares the results. It
    then prints the fraction of hits and rates for both functions.

    Args:
        Nrep (int): Number of repetitions for the test.
        Nbvh (int): Number of BVH structures to generate.
        Nsamp (int): Number of samples per repetition.
    """

    Nrep, Nbvh, Nsamp = 1, 1_000, 10000
    N = Nrep * Nsamp
    tbvh, tnai = 0, 0
    nhit, ntot = 0, 0
    for i in range(Nrep):
        pts_bvh = np.random.rand(Nbvh, 7)
        bvh = create_bvh7d(pts_bvh)
        assert np.allclose(bvh.com(), np.mean(pts_bvh, axis=0))

        mindis = 0.2
        samp = np.random.rand(Nsamp, 7)
        tbvh0 = perf_counter()
        isect1 = 0 <= bvh_isect7d(bvh, samp, mindis)
        tbvh += perf_counter() - tbvh0
        tnai0 = perf_counter()
        isect2 = 0 <= bvh_isect7d_naive(bvh, samp, mindis)
        tnai += perf_counter() - tnai0
        assert np.all(isect1 == isect2)
        nhit += np.sum(isect1)
        ntot += Nsamp

    print(
        f' bvh_isect7d frchit: {nhit / ntot:4.3f}',
        f'bvh rate: {int(N / tbvh):,} naive rate: {int(N / tnai):,}',
    )


def test_bvh_mindist4():
    """Test the performance of BVH-based mindistance calculations.

    This function benchmarks the `bvh_mindist4d` and `bvh_mindist4d_naive`
    functions by comparing their performance in finding the minimum distance
    between a set of points and a BVH structure. It measures execution time
    for both methods and prints out the results, including the minimum
    distance found and the rates at which each method completes.
    """

    Nrep, Nbvh, Nsamp = 1, 1_000, 1000
    N = Nrep * Nsamp
    tbvh, tnai = 0, 0
    mindis = 9e9
    for i in range(Nrep):
        pts_bvh = np.random.rand(Nbvh, 4)
        bvh = create_bvh4d(pts_bvh)
        assert np.allclose(bvh.com(), np.mean(pts_bvh, axis=0))

        samp = np.random.rand(Nsamp, 4) + [0.2, 0.4, 0, 0]
        tbvh0 = perf_counter()
        mindist1, w1 = bvh_mindist4d(bvh, samp)
        tbvh += perf_counter() - tbvh0
        tnai0 = perf_counter()
        mindist2, w2 = bvh_mindist4d_naive(bvh, samp)
        tnai += perf_counter() - tnai0
        assert np.allclose(mindist1, mindist2)
        assert np.all(w1 == w2)
        mindis = min(mindis, np.min(mindist1))

    print(
        f' bvh_mindist4d mind: {mindis:5.3f} bvh rate: {int(N / tbvh):,} naive rate: {int(N / tnai):,}',
        # f"{tnai/tbvh:7.3f}",
    )
    print('bvh', tbvh)
    print('nai', tnai)


def test_bvh_mindist7():
    """Test the BVH mindist7d function against a naive implementation.

    This function runs a series of tests to compare the performance and
    accuracy of the `bvh_mindist7d` function with its naive counterpart. It
    measures the time taken by both functions and prints the results,
    including the minimum distance found, the rate of execution for each
    method, and the total time taken by each.
    """

    Nrep, Nbvh, Nsamp = 1, 1_000, 10000
    N = Nrep * Nsamp
    tbvh, tnai = 0, 0
    mindis = 9e9
    for i in range(Nrep):
        pts_bvh = np.random.rand(Nbvh, 7)
        bvh = create_bvh7d(pts_bvh)
        assert np.allclose(bvh.com(), np.mean(pts_bvh, axis=0))

        samp = np.random.rand(Nsamp, 7) + [0, 0, 0, 0, 0, 0, 0]
        tbvh0 = perf_counter()
        mindist1, w1 = bvh_mindist7d(bvh, samp)
        tbvh += perf_counter() - tbvh0
        tnai0 = perf_counter()
        mindist2, w2 = bvh_mindist7d_naive(bvh, samp)
        tnai += perf_counter() - tnai0
        assert np.allclose(mindist1, mindist2)
        assert np.all(w1 == w2)
        mindis = min(mindis, np.min(mindist1))

    print(
        f' bvh_mindist7d mind: {mindis:5.3f} bvh rate: {int(N / tbvh):,} naive rate: {int(N / tnai):,}',
        # f"{tnai/tbvh:7.3f}",
    )
    print('bvh', tbvh)
    print('nai', tnai)


if __name__ == '__main__':
    # print(4096 / 2.9, 1000000 / 38.3)
    # test_bvh_isect7()
    # test_bvh_isect7()
    test_bvh_mindist7()
    # test_bvh_mindist4()
