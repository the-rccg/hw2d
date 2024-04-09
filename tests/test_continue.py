import pytest
import os

from hw2d.run import run as run
from hw2d.run2 import run as run2


def test_continue_run2(tmp_path=os.getcwd()):
    test_parameters = dict(
        # Physics & Numerics
        step_size=0.025,
        grid_pts=256,
        k0=0.15,
        N=3,
        nu=1.0e-08,
        c1=1,
        kappa_coeff=1.0,
        poisson_bracket_coeff=1.0,
        # Run
        show_property="gamma_c",
        # Initialization
        seed=None,
        init_type="normal",
        init_scale=1 / 100,
        # Saving
        output_path=f"{tmp_path}/test_continue.h5",
        buffer_length=2,
        snaps=1,
        chunk_size=1,
        downsample_factor=4,
        # Movie
        movie=False,
        # Properties
        properties=["gamma_c"],
        # Plotting
        plot_properties=[],
        # Other
        debug=False,
    )
    run2(continue_file=False, force_recompute=True, end_time=1, **test_parameters)
    run2(continue_file=True, force_recompute=False, end_time=2, **test_parameters)


def test_continue_run(tmp_path=os.getcwd()):
    test_parameters = dict(
        # Physics & Numerics
        step_size=0.025,
        grid_pts=64,
        k0=0.15,
        N=3,
        nu=1.0e-08,
        c1=1,
        kappa_coeff=1.0,
        poisson_bracket_coeff=1.0,
        # Initialization
        seed=None,
        init_type="normal",
        init_scale=1 / 100,
        # Saving
        output_path=f"{tmp_path}/test_continue.h5",
        buffer_length=2,
        snaps=1,
        chunk_size=1,
        downsample_factor=2,
        # Movie
        movie=False,
        # Plotting
        plot_properties=[],
        # Other
        debug=False,
    )
    run(continue_file=False, force_recompute=True, end_time=1, **test_parameters)
    run(continue_file=True, force_recompute=False, end_time=2, **test_parameters)


if __name__ == "__main__":
    #pytest.main()
    test_continue_run(os.getcwd())
    test_continue_run2(os.getcwd())
