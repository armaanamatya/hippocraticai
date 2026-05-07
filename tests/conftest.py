"""Test isolation — each test gets its own SQLite trace DB so the project's
real `traces.db` is never touched by the test suite. Tests that don't care
about tracing inherit the isolated path; tests that DO want tracing get a
clean fresh DB per test."""
import pytest


@pytest.fixture(autouse=True)
def _isolated_trace_db(tmp_path, monkeypatch):
    monkeypatch.setenv("STORY_DB_PATH", str(tmp_path / "trace_test.db"))
