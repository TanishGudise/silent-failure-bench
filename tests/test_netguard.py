import importlib.util
import socket
from pathlib import Path


def _load(monkeypatch, hosts):
    monkeypatch.setenv("NETGUARD_ALLOW_HOSTS", hosts)
    spec = importlib.util.spec_from_file_location("ng", Path(__file__).parent.parent / "tools/netguard/sitecustomize.py")
    original = (socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    socket.getaddrinfo, socket.socket.connect, socket.socket.connect_ex = original
    return module


def test_only_addresses_of_allowed_hosts_are_reachable(monkeypatch):
    ng = _load(monkeypatch, "api.example.test")
    monkeypatch.setattr(ng, "_orig_gai", lambda host, *a, **k: [(2, 1, 6, "", ("192.0.2.7", 443))])
    assert ng.allowed_address(("127.0.0.1", 80))
    assert not ng.allowed_address(("192.0.2.7", 443))
    ng.getaddrinfo("api.example.test", 443)
    assert ng.allowed_address(("192.0.2.7", 443))
    assert not ng.allowed_address(("192.0.2.8", 443))
    try:
        ng.getaddrinfo("evil.example.test", 443)
        raise AssertionError("resolved a host that isn't allowed")
    except socket.gaierror:
        pass
