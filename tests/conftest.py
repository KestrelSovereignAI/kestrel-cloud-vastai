"""Test configuration for kestrel-cloud-vastai.

Adds --run-cloud opt-in for cloud_resource-marked tests, mirroring
the framework's convention. Without --run-cloud, real-cloud tests
skip cleanly.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-cloud",
        action="store_true",
        default=False,
        help="Run tests that touch real cloud (requires Vast.ai creds)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip cloud_resource tests unless --run-cloud is provided."""
    if config.getoption("--run-cloud"):
        return
    skip_cloud = pytest.mark.skip(reason="needs --run-cloud option to run")
    for item in items:
        if "cloud_resource" in item.keywords:
            item.add_marker(skip_cloud)
