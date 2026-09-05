"""
Walks a hash-chained audit log end to end and confirms the chain is
unbroken -- detects tampering (a record's own hash no longer matches its
content) and deletion/reordering (a record's prev_hash no longer matches
the actual previous record's hash), without trusting anything about how
the log file itself was stored on disk.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .log import GENESIS_HASH, _canonical


@dataclass(frozen=True)
class VerifyResult:
    intact: bool
    record_count: int
    broken_at_line: int | None
    reason: str | None


def verify_log(path: Path) -> VerifyResult:
    if not path.exists():
        return VerifyResult(True, 0, None, None)  # nothing logged yet is trivially intact

    prev_hash = GENESIS_HASH
    count = 0
    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if record.get("prev_hash") != prev_hash:
                return VerifyResult(
                    False, count, line_no,
                    f"prev_hash mismatch at line {line_no}: expected {prev_hash}, got {record.get('prev_hash')}",
                )

            body = {"kind": record["kind"], "at": record["at"], "prev_hash": record["prev_hash"], "payload": record["payload"]}
            expected_hash = hashlib.sha256(_canonical(body)).hexdigest()
            if expected_hash != record.get("record_hash"):
                return VerifyResult(False, count, line_no, f"record_hash mismatch at line {line_no}: content does not match its recorded hash")

            prev_hash = record["record_hash"]
            count += 1

    return VerifyResult(True, count, None, None)
