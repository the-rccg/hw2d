import os
import numpy as np
import h5py

from hw2d.run import run


def test_run(tmp_path):
    run(grid_pts=64, c1=1.0, end_time=0.5, movie=False, output_path=f"{tmp_path}/test.h5")


def test_recording_start_time(tmp_path=os.getcwd()):
    grid_pts = 64
    end_time = 1
    recording_start_time = 0.5
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
        recording_start_time=recording_start_time,
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
        print(dict(hf.attrs).get("initial_time", 0))
        print(recording_start_time)
        print(hf["density"].shape, np.array([int(end_time - dict(hf.attrs).get("initial_time", 0) // test_parameters["step_size"]) + 1, grid_pts // test_parameters["downsample_factor"], grid_pts // test_parameters["downsample_factor"]]) )
        assert hf["density"].shape[0] == (int((end_time - dict(hf.attrs).get("initial_time", 0)) / test_parameters["step_size"])) + 1
        assert np.array_equal(hf["density"].shape[1:], np.array([grid_pts, grid_pts]) // test_parameters["downsample_factor"]), f"{hf['density'].shape[1:]} - {[grid_pts, grid_pts]}"
        assert hf["gamma_n"][-1] != 0


if __name__ == "__main__":
    test_run(os.getcwd())
    test_recording_start_time(os.getcwd())
