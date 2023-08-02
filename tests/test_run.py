from hw2d.run import run


def test_run(tmp_path):
    run(end_time=0.5, movie=False, output_path=tmp_path / "test.h5")
