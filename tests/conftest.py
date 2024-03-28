import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return

    skip_slow = pytest.mark.skip(reason="need --full option to run")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
