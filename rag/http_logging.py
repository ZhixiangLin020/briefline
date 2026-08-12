"""Small helpers for logging HTTP request metadata without credentials."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def redact_query_parameter(url: str, parameter_name: str) -> str:
    """Return *url* with every matching query-parameter value redacted."""

    parts = urlsplit(str(url))
    query = [
        (name, "***" if name == parameter_name else value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment)
    )
