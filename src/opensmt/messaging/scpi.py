from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class SCPIKind(str, Enum):
    QUERY = "query"
    SET = "set"
    RESPONSE = "response"
    WORKING = "working"
    ACTION = "action"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class SCPIMessage:
    kind: SCPIKind
    command: str
    value: Any = None
    raw: str = ""


_QUERY_RE = re.compile(r"^(?P<cmd>:[A-Za-z0-9_:\-]+)\?$")
_RESPONSE_RE = re.compile(r"^(?P<cmd>:[A-Za-z0-9_:\-]+)\?\s+(?P<value>.+)$")
_SET_RE = re.compile(r"^(?P<cmd>:[A-Za-z0-9_:\-]+)\s+(?P<value>.+)$")
_WORKING_RE = re.compile(r"^(?P<cmd>:[A-Za-z0-9_:\-]+)\s+WORKING$")
_ACTION_RE = re.compile(r"^(?P<cmd>:[A-Za-z0-9_:\-]+)$")


def parse_value(value: str) -> Any:
    token = value.strip()
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return token[1:-1]

    try:
        if any(ch in token for ch in (".", "e", "E")):
            return float(token)
        return int(token)
    except ValueError:
        return token


def render_value(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


def normalize_command(command: str) -> str:
    cmd = command.strip().upper()
    if not cmd.startswith(":"):
        cmd = f":{cmd}"
    return cmd


def parse_scpi(text: str) -> SCPIMessage:
    raw = text.strip()

    m = _WORKING_RE.match(raw)
    if m:
        return SCPIMessage(
            kind=SCPIKind.WORKING,
            command=normalize_command(m.group("cmd")),
            raw=raw,
        )

    m = _RESPONSE_RE.match(raw)
    if m:
        return SCPIMessage(
            kind=SCPIKind.RESPONSE,
            command=normalize_command(m.group("cmd")),
            value=parse_value(m.group("value")),
            raw=raw,
        )

    m = _QUERY_RE.match(raw)
    if m:
        return SCPIMessage(
            kind=SCPIKind.QUERY,
            command=normalize_command(m.group("cmd")),
            raw=raw,
        )

    m = _SET_RE.match(raw)
    if m:
        return SCPIMessage(
            kind=SCPIKind.SET,
            command=normalize_command(m.group("cmd")),
            value=parse_value(m.group("value")),
            raw=raw,
        )

    m = _ACTION_RE.match(raw)
    if m:
        return SCPIMessage(
            kind=SCPIKind.ACTION,
            command=normalize_command(m.group("cmd")),
            raw=raw,
        )

    return SCPIMessage(kind=SCPIKind.UNKNOWN, command="", raw=raw)
