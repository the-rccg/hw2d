import pytest
from os.path import dirname, realpath
import h5py

from hw2d.utils.plot.image import plot_dict
from hw2d.utils.plot.movie import get_extended_viridis


def test_plot_dict():
    reference = h5py.File(
        f"{dirname(dirname(dirname(realpath(__file__))))}/reference.h5",
        "r",
    )
    test_dict = {name: reference[name][0] for name in ("density", "omega", "phi")}
    ve = get_extended_viridis(vals=600)

    fig = plot_dict(
        test_dict,
        cmap=ve,
        couple_cbars=False,
        figsize=None,
        sharex=True,
        sharey=True,
        vertical=False,
        cbar_label_spacing=1.7,
    )


if __name__ == "__main__":
    test_plot_dict()
