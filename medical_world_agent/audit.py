from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .sanitizer import SanitizerConfig, sanitize_audit_payload


class AuditLogger:
    def __init__(
        self,
        log_path: str | Path = "reports/audit_log.jsonl",
        sanitizer_config: SanitizerConfig | None = None,
    ) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._sanitizer_config = sanitizer_config or SanitizerConfig()

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        sanitized = sanitize_audit_payload(payload, self._sanitizer_config)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": sanitized,
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def log_path(self) -> Path:
        return self._log_path
