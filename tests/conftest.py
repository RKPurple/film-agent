"""
Shared fixtures for the agent-loop tests.

- Network is blocked for the whole session (from pytest_configure, so
  module imports are covered too): any real socket connection or DNS lookup
  raises. The tests use a fake Gemini client and fake tools and must never
  touch the network.
- The real logs/agent_traces.jsonl must never be written: tests pass a
  tmp log_path to with_trace_log(), and a session-level check fails the run
  if the real file's size or mtime changed.
"""

import os
import socket

import pytest


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise NetworkBlocked("network access attempted during tests")


def pytest_configure(config):
    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked


@pytest.fixture(scope="session", autouse=True)
def real_trace_log_untouched():
    from cinemagent.agent import TRACE_LOG_PATH

    def snapshot():
        return (os.path.getsize(TRACE_LOG_PATH), os.path.getmtime(TRACE_LOG_PATH)) if TRACE_LOG_PATH.exists() else None

    before = snapshot()
    yield
    assert snapshot() == before, f"tests wrote to the real trace log {TRACE_LOG_PATH}"


@pytest.fixture
def fake_tools(monkeypatch):
    """Replace the agent's tool dispatch with scripted results. There's no
    injection seam on run_agent(), so this patches cinemagent.agent's
    imported call_tool name."""
    from fakes import FakeTools

    tools = FakeTools()
    monkeypatch.setattr("cinemagent.agent.call_tool", tools)
    return tools


@pytest.fixture
def trace_path(tmp_path):
    return tmp_path / "agent_traces.jsonl"
