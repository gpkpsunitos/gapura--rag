from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.config import Settings


@dataclass(frozen=True, slots=True)
class AccountRateLimitDecision:
    allowed: bool
    status_code: int = 200
    error: str = ""
    retry_after_seconds: int = 0
    remaining: int | None = None


def extract_bearer_token(authorization_header: str | None) -> str:
    if not authorization_header:
        return ""

    prefix = "bearer "
    if authorization_header.lower().startswith(prefix):
        return authorization_header[len(prefix) :].strip()

    return ""


def consume_account_rate_limit(
    access_token: str,
    settings: Settings,
) -> AccountRateLimitDecision:
    if not settings.account_rate_limit_enabled:
        return AccountRateLimitDecision(allowed=True)

    if not access_token:
        return AccountRateLimitDecision(
            allowed=False,
            status_code=401,
            error="Virtual Assistant login is required.",
        )

    if (
        not settings.account_rate_limit_consume_url
        or not settings.account_rate_limit_internal_secret
    ):
        raise EnvironmentError(
            "ACCOUNT_RATE_LIMIT_CONSUME_URL and "
            "ACCOUNT_RATE_LIMIT_INTERNAL_SECRET must be configured"
        )

    request = urllib.request.Request(
        settings.account_rate_limit_consume_url,
        data=json.dumps({"token": access_token}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.account_rate_limit_internal_secret}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = _read_json(response.read())
            return AccountRateLimitDecision(
                allowed=bool(payload.get("allowed", True)),
                remaining=_optional_int(payload.get("remaining")),
                retry_after_seconds=_optional_int(
                    payload.get("retry_after_seconds")
                )
                or 0,
            )
    except urllib.error.HTTPError as exc:
        payload = _read_json(exc.read())
        retry_after = _optional_int(
            exc.headers.get("Retry-After")
            or payload.get("retry_after_seconds")
        ) or 0
        return AccountRateLimitDecision(
            allowed=False,
            status_code=exc.code,
            error=str(payload.get("error") or "Virtual Assistant quota check failed."),
            retry_after_seconds=retry_after,
            remaining=_optional_int(payload.get("remaining")),
        )
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Virtual Assistant quota check failed: {exc}") from exc


def _read_json(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}

    try:
        decoded = raw.decode("utf-8")
        payload = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
