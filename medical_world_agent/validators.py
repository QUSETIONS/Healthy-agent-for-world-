from __future__ import annotations

import ipaddress
from pathlib import Path
import re
from socket import getaddrinfo, AF_INET, AF_INET6
from typing import Any
from urllib.parse import urlparse


_ALLOWED_URL_SCHEMES = {"http", "https"}
_ALLOWED_CORPUS_ROOTS = ("data", "cases", "reports")

_PRIVATE_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


class ValidationError(Exception):
    pass


def _is_private_ip(host: str) -> bool:
    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _PRIVATE_NETWORKS)
    except ValueError:
        pass

    try:
        results = getaddrinfo(host, None, AF_INET)
        results += getaddrinfo(host, None, AF_INET6)
    except OSError:
        return _looks_like_ip_literal(host)

    for family, _type, _proto, _canon, sockaddr in results:
        try:
            addr = ipaddress.ip_address(sockaddr[0])
            if any(addr in net for net in _PRIVATE_NETWORKS):
                return True
        except ValueError:
            continue
    return False


def _looks_like_ip_literal(host: str) -> bool:
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host)) or ":" in host


def validate_url(url: str) -> str:
    if not url or not url.strip():
        raise ValidationError("URL must not be empty")

    parsed = urlparse(url.strip())

    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValidationError(
            f"URL scheme '{parsed.scheme}' is not allowed. "
            f"Only {sorted(_ALLOWED_URL_SCHEMES)} are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValidationError("URL must contain a valid hostname")

    if _is_private_ip(hostname):
        raise ValidationError("URLs pointing to private/internal networks are blocked")

    return url.strip()


def validate_corpus_path(path: str) -> str:
    if not path or not path.strip():
        raise ValidationError("Corpus path must not be empty")

    cleaned = path.strip()

    if ".." in cleaned:
        raise ValidationError("Path traversal sequences ('..') are not allowed")

    dangerous_patterns = ["~", "$", "%", "\x00"]
    for pat in dangerous_patterns:
        if pat in cleaned:
            raise ValidationError(f"Path contains forbidden character: '{pat}'")

    if not cleaned.endswith(".json"):
        raise ValidationError("Corpus path must point to a .json file")

    path_obj = Path(cleaned)
    if path_obj.is_absolute():
        raise ValidationError("Absolute corpus paths are not allowed")

    normalized = cleaned.replace("\\", "/")
    top_level = normalized.split("/", 1)[0]
    if top_level not in _ALLOWED_CORPUS_ROOTS:
        raise ValidationError(
            f"Corpus path must be under allowed roots: {', '.join(_ALLOWED_CORPUS_ROOTS)}"
        )

    return cleaned


def validate_message_length(message: str, max_length: int = 2000) -> str:
    if len(message) > max_length:
        raise ValidationError(
            f"Message length {len(message)} exceeds maximum {max_length}"
        )
    return message


def validate_noise_range(value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValidationError(
            f"observation_noise must be between 0.0 and 1.0, got {value}"
        )
    return value


def validate_evidence_top_k(value: int) -> int:
    if not (1 <= value <= 20):
        raise ValidationError(
            f"evidence_top_k must be between 1 and 20, got {value}"
        )
    return value
