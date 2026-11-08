import numpy as np
import pytest

from LebwohlLasher import initdat, all_energy, all_energy_numpy


@pytest.mark.parametrize("arr, nmax", [
    (initdat(3), 3),
    (initdat(5), 5),
    (initdat(12), 12),
])
def test_all_energy_numpy(arr, nmax):
    output = all_energy_numpy(arr)

    assert np.isclose(output, all_energy(arr, nmax))