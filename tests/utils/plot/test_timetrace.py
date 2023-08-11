import pytest
import h5py
from os.path import dirname, realpath

from hw2d.utils.plot.timetrace import plot_timetraces


def test_timetrace():
    file_path = f"{dirname(dirname(dirname(realpath(__file__))))}/traces.h5"
    plot_timetraces(
        file_path=file_path,
        out_path=None,
        properties=("gamma_n", "gamma_c"),
        t0=0,
        t0_std=0,
    )


if __name__ == "__main__":
    test_timetrace()
