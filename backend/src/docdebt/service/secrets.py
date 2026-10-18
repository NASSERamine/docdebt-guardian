"""Encryption of the API keys saved from the application.

A key typed on the Model page is stored encrypted (Fernet: AES-128-CBC with
an HMAC-SHA256 tag) and is never sent back to the browser, which only sees
its last four characters. The encryption key comes from
``DOCDEBT_SECRET_KEY``; without it, one is generated on first start next to
the state database (``secret.key``, readable by the owner only), so a local
installation works out of the box. Losing that file makes the saved API keys
unreadable: they must then be typed again.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import logging
import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("docdebt.service")


class SecretError(RuntimeError):
    """A saved secret cannot be decrypted (the encryption key changed)."""


def _fernet_key(material: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(material.encode("utf-8")).digest())


def load_cipher(state_dir: str | Path) -> Fernet:
    """The cipher for saved secrets: from ``DOCDEBT_SECRET_KEY``, else from
    ``<state_dir>/secret.key`` (created on first use)."""
    material = os.environ.get("DOCDEBT_SECRET_KEY")
    if not material:
        path = Path(state_dir) / "secret.key"
        if path.is_file():
            material = path.read_text(encoding="utf-8").strip()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            material = secrets.token_urlsafe(32)
            path.write_text(material, encoding="utf-8")
            with contextlib.suppress(OSError):
                path.chmod(0o600)
            logger.warning(
                "DOCDEBT_SECRET_KEY is not set: generated %s to encrypt saved API keys. "
                "Back it up, or set DOCDEBT_SECRET_KEY.",
                path,
            )
    return Fernet(_fernet_key(material))


def encrypt(cipher: Fernet, value: str) -> str:
    return cipher.encrypt(value.encode("utf-8")).decode("ascii")


def decrypt(cipher: Fernet, token: str) -> str:
    try:
        return cipher.decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise SecretError("The saved API key cannot be decrypted: type it again on the Model page.") from exc


def key_hint(value: str) -> str:
    """What the application shows of a key: its last four characters."""
    return f"…{value[-4:]}" if len(value) >= 8 else "…"
