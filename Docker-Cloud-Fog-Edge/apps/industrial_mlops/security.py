from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import hmac
import json
import os
import tempfile
from typing import Any


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    body = canonical_json({k: v for k, v in payload.items() if k != "signature"})
    return hmac.new(secret.encode("utf-8"), body.encode("utf-8"), sha256).hexdigest()


def verify_payload(payload: dict[str, Any], secret: str) -> bool:
    signature = payload.get("signature")
    if not signature:
        return False
    expected = sign_payload(payload, secret)
    return hmac.compare_digest(signature, expected)


def compute_bytes_digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def compute_file_digest(path: Path) -> str:
    return compute_bytes_digest(path.read_bytes())


def compute_directory_digest(path: Path) -> str:
    digest = sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
