"""Shared pytest configuration.

Registers the ``live`` marker used by tests that hit a real public API. Those tests
are skipped unless ``ENGINE_LIVE_TESTS`` is set, and only ever send public textbook
variants. Registration is guarded so that a second registration (another conftest,
an ini option) is harmless.
"""


def pytest_configure(config):
    try:
        config.addinivalue_line("markers", "live: hits a real public API; needs ENGINE_LIVE_TESTS=1")
    except Exception:  # already registered elsewhere
        pass
