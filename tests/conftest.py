"""pytest configuration for pUniFind end-to-end tests."""


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (requires GPU)")
