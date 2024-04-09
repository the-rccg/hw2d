import os

from hw2d.run import run


def test_run(tmp_path):
    run(grid_pts=64, c1=1.0, end_time=0.5, movie=False, output_path=f"{tmp_path}/test.h5")


if __name__ == "__main__":
    test_run(os.getcwd())
