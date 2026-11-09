import pytest
import numpy as np

import LebwohlLasher as l
import LebwohlLasher_pyx as lpyx

@pytest.mark.parametrize("arr, ix, iy, nmax", [
    (np.zeros((3,3)), 0, 0, 3),
    (lpyx.initdat(3), 0, 0, 3),
    (lpyx.initdat(12), 0, 0, 12)
])
def test_one_energy(arr, ix, iy, nmax):
    output = lpyx.one_energy_wrapper(arr, ix, iy, nmax)

    assert np.isclose(output, l.one_energy(arr, ix, iy, nmax))


@pytest.mark.parametrize("arr, nmax", [
    (lpyx.initdat(3), 3),
    (lpyx.initdat(12), 12)
])
def test_all_energy(arr, nmax):
    output = lpyx.all_energy_wrapper(arr, nmax, 1)

    assert np.isclose(output, l.all_energy(arr, nmax))


@pytest.mark.parametrize("arr, nmax", [
    (lpyx.initdat(3), 3),
    (lpyx.initdat(12), 12)
])
def test_get_order(arr, nmax):
    output = lpyx.get_order_wrapper(arr, nmax, 1)

    assert np.isclose(output, l.get_order(arr, nmax))


@pytest.mark.parametrize("arr, Ts, nmax", [
    (lpyx.initdat(3), 0.9, 3),
    (lpyx.initdat(5), 0.9, 5)
])
def test_MC_step(arr, Ts, nmax):
    outputs = []
    expects = []

    for n in range(10000):
        outputs.append(lpyx.MC_step(arr, Ts, nmax))
        expects.append(l.MC_step(arr, Ts, nmax))

    assert abs(np.mean(outputs)-np.mean(expects)) <= 0.015