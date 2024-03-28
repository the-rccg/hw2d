import pytest


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return

    skip_slow = pytest.mark.skip(reason="need --all option to run")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
