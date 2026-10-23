"""Network guard for every Python process in this repo.

Loopback is always allowed. NETGUARD_ALLOW_HOSTS (comma separated) adds hosts;
connections are then allowed only to addresses those names resolved to in this
process. Everything else fails closed and is logged to NETGUARD_LOG.
"""
import ipaddress
import os
import socket
import threading
import time

_LOG = os.environ.get("NETGUARD_LOG")
_ALLOW_HOSTS = {h.strip().lower() for h in os.environ.get("NETGUARD_ALLOW_HOSTS", "").split(",") if h.strip()}
_ALLOWED_IPS = set()
_LOCK = threading.Lock()


def _log(kind, target):
    if _LOG:
        with open(_LOG, "a") as handle:
            handle.write(f"{time.time():.3f} pid={os.getpid()} {kind} {target!r}\n")


def _is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in ("localhost", "")


_orig_gai = socket.getaddrinfo


def getaddrinfo(host, *args, **kwargs):
    name = (host.decode() if isinstance(host, bytes) else (host or "")).lower()
    if _is_loopback(name):
        return _orig_gai(host, *args, **kwargs)
    if name not in _ALLOW_HOSTS:
        _log("BLOCK_DNS", name)
        raise socket.gaierror(socket.EAI_NONAME, f"netguard: {name} not allowed")
    results = _orig_gai(host, *args, **kwargs)
    with _LOCK:
        _ALLOWED_IPS.update(r[4][0] for r in results)
    return results


def allowed_address(address):
    host = address[0]
    with _LOCK:
        return _is_loopback(host) or host in _ALLOWED_IPS


socket.getaddrinfo = getaddrinfo
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _check(sock, address):
    if sock.family in (socket.AF_INET, socket.AF_INET6) and not allowed_address(address):
        _log("BLOCK_CONNECT", address)
        raise ConnectionRefusedError(f"netguard: {address} not allowed")


def connect(self, address):
    _check(self, address)
    return _orig_connect(self, address)


def connect_ex(self, address):
    _check(self, address)
    return _orig_connect_ex(self, address)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
