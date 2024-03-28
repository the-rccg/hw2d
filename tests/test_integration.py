import pytest
from hw2d.run import run


@pytest.mark.slow()
def test_integration(tmp_path):
    run(
        step_size=0.01,
        end_time=1,
        grid_pts=512,
        k0=0.15,
        N=3,
        nu=1e-9,
        kappa_coeff=1,
        movie=False,
        debug=True,
        properties=[],
        plot_properties=[],
        output_path=tmp_path / "test.h5",
    )
