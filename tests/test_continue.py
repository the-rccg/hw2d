import pytest
import os
import h5py
import numpy as np

from hw2d.run import run as run
from hw2d.run2 import run as run2


def test_continue_run(tmp_path=os.getcwd()):
    grid_pts = 64
    end_time = 1
    test_parameters = dict(
        # Physics & Numerics
        step_size=0.025,
        grid_pts=grid_pts,
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
    # Original run
    run(continue_file=False, force_recompute=True, end_time=end_time, **test_parameters)
    # Verify Data
    with h5py.File(f"{tmp_path}/test_continue.h5") as hf:
        print(hf["density"].shape, np.array([int(end_time / test_parameters["step_size"]) + 1, grid_pts // test_parameters["downsample_factor"], grid_pts // test_parameters["downsample_factor"]]) )
        assert hf["density"].shape[0] == end_time / test_parameters["step_size"] + 1
        assert np.array_equal(hf["density"].shape[1:], np.array([grid_pts, grid_pts]) // test_parameters["downsample_factor"]), f"{hf['density'].shape[1:]} - { [grid_pts, grid_pts]}"
        assert hf["gamma_n"][-1] != 0
    # Continue
    end_time = end_time + 1
    run(continue_file=True, force_recompute=False, end_time=end_time, **test_parameters)
    # Verify Data
    with h5py.File(f"{tmp_path}/test_continue.h5") as hf:
        print(hf["density"], np.array([int(end_time / test_parameters["step_size"]) + 1, grid_pts // test_parameters["downsample_factor"], grid_pts // test_parameters["downsample_factor"]]) )
        assert hf["density"].shape[0] == int(end_time / test_parameters["step_size"]) + 1, f"{hf['density'].shape[0]} vs {int(end_time / test_parameters['step_size']) + 1}"
        assert np.array_equal(hf["density"].shape[1:], np.array([grid_pts, grid_pts]) // test_parameters["downsample_factor"]), f"{hf['density'].shape[1:]} - {[grid_pts, grid_pts]}"
        assert hf["gamma_n"][-1] != 0


def test_continue_run2(tmp_path=os.getcwd()):
    grid_pts = 256
    end_time = 1
    test_parameters = dict(
        # Physics & Numerics
        step_size=0.025,
        grid_pts=grid_pts,
        k0=0.15,
        N=3,
        nu=1.0e-08,
        c1=1,
        kappa_coeff=1.0,
        poisson_bracket_coeff=1.0,
        # Run
        show_property="gamma_n",
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
        properties=["gamma_n"],
        # Plotting
        plot_properties=[],
        # Other
        debug=False,
    )
    run2(continue_file=False, force_recompute=True, end_time=end_time, **test_parameters)
    # Verify Data
    with h5py.File(f"{tmp_path}/test_continue.h5") as hf:
        print(hf["density"], np.array([int(end_time / test_parameters["step_size"]) + 1, grid_pts // test_parameters["downsample_factor"], grid_pts // test_parameters["downsample_factor"]]) )
        assert hf["density"].shape[0] == end_time / test_parameters["step_size"] + 1
        assert np.array_equal(hf["density"].shape[1:], np.array([grid_pts, grid_pts]) // test_parameters["downsample_factor"]), f"{hf['density'].shape[1:]} - { [grid_pts, grid_pts]}"
        assert hf["gamma_n"][-1] != 0
    # Continue
    end_time = end_time + 1
    run2(continue_file=True, force_recompute=False, end_time=end_time, **test_parameters)
    # Verify Data
    with h5py.File(f"{tmp_path}/test_continue.h5") as hf:
        print(hf["density"], np.array([int(end_time / test_parameters["step_size"]) + 1, grid_pts // test_parameters["downsample_factor"], grid_pts // test_parameters["downsample_factor"]]) )
        assert hf["density"].shape[0] == end_time / test_parameters["step_size"] + 1
        assert np.array_equal(hf["density"].shape[1:], np.array([grid_pts, grid_pts]) // test_parameters["downsample_factor"]), f"{hf['density'].shape[1:]} - { [grid_pts, grid_pts]}"
        assert hf["gamma_n"][-1] != 0
    # Continue
    end_time = end_time + 1


if __name__ == "__main__":
    #pytest.main()
    test_continue_run(os.getcwd())
    test_continue_run2(os.getcwd())
