"""
src/collection/queue.py

Work queue for deferred data collection tasks.
State persisted in data/queue/pending.json and data/queue/completed.json.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import structlog

log = structlog.get_logger()

TaskType = Literal["price_fetch", "price_backfill", "options_chain", "ticker_details", "financials", "split_correction"]
Priority = Literal["high", "normal", "low"]
TaskStatus = Literal["pending", "ready", "completed", "failed", "expired"]

_PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}


@dataclass
class QueueItem:
    task_id: str
    task_type: TaskType
    ticker: str
    requested_date: str          # YYYY-MM-DD
    requested_by: str
    priority: Priority = "normal"
    status: TaskStatus = "pending"
    created_at: str = field(default_factory=lambda: _now_iso())
    completed_at: str | None = None
    retry_count: int = 0
    data_path: str | None = None
    error_msg: str | None = None
    metadata: dict = field(default_factory=dict)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dedup_key(item: QueueItem) -> tuple:
    return (item.task_type, item.ticker, item.requested_date)


class WorkQueue:
    def __init__(self, queue_dir: str | Path = "data/queue"):
        self._dir = Path(queue_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._pending_path = self._dir / "pending.json"
        self._completed_path = self._dir / "completed.json"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_pending(self) -> list[QueueItem]:
        if not self._pending_path.exists():
            return []
        with open(self._pending_path) as f:
            raw = json.load(f)
        return [QueueItem(**r) for r in raw]

    def _save_pending(self, items: list[QueueItem]) -> None:
        with open(self._pending_path, "w") as f:
            json.dump([asdict(i) for i in items], f, indent=2)

    def _load_completed(self) -> list[dict]:
        if not self._completed_path.exists():
            return []
        with open(self._completed_path) as f:
            return json.load(f)

    def _save_completed(self, records: list[dict]) -> None:
        with open(self._completed_path, "w") as f:
            json.dump(records, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        task_type: TaskType,
        ticker: str,
        requested_date: str,
        requested_by: str,
        priority: Priority = "normal",
        metadata: dict | None = None,
    ) -> str:
        """
        Add a task to the queue. Deduplicates by (task_type, ticker, requested_date).
        Returns the task_id (existing or new).
        """
        items = self._load_pending()

        # Check for existing pending/ready task with same key
        for item in items:
            if _dedup_key(item) == (task_type, ticker, requested_date) and item.status in ("pending", "ready"):
                log.debug("queue.deduplicated", task_type=task_type, ticker=ticker, date=requested_date)
                return item.task_id

        task_id = str(uuid.uuid4())
        new_item = QueueItem(
            task_id=task_id,
            task_type=task_type,
            ticker=ticker,
            requested_date=requested_date,
            requested_by=requested_by,
            priority=priority,
            metadata=metadata or {},
        )
        items.append(new_item)
        self._save_pending(items)
        log.info("queue.enqueued", task_id=task_id, task_type=task_type, ticker=ticker, priority=priority)
        return task_id

    def get_pending(
        self,
        task_type: TaskType | None = None,
        priority: Priority | None = None,
    ) -> list[QueueItem]:
        """Return pending tasks, optionally filtered, sorted high→normal→low."""
        items = self._load_pending()
        result = [i for i in items if i.status in ("pending", "ready")]
        if task_type:
            result = [i for i in result if i.task_type == task_type]
        if priority:
            result = [i for i in result if i.priority == priority]
        result.sort(key=lambda i: _PRIORITY_ORDER.get(i.priority, 99))
        return result

    def mark_complete(self, task_id: str, data_path: str | None = None) -> None:
        items = self._load_pending()
        completed = self._load_completed()

        for item in items:
            if item.task_id == task_id:
                item.status = "completed"
                item.completed_at = _now_iso()
                item.data_path = data_path
                completed.append(asdict(item))
                log.info("queue.completed", task_id=task_id, data_path=data_path)
                break

        # Remove from pending
        items = [i for i in items if i.task_id != task_id or i.status != "completed"]
        self._save_pending(items)
        self._save_completed(completed)

    def mark_failed(self, task_id: str, error_msg: str) -> None:
        items = self._load_pending()
        for item in items:
            if item.task_id == task_id:
                item.retry_count += 1
                item.error_msg = error_msg
                if item.retry_count >= 5:
                    item.status = "failed"
                    log.warning("queue.failed_permanently", task_id=task_id, retries=item.retry_count)
                else:
                    log.warning("queue.failed_retryable", task_id=task_id, retries=item.retry_count)
                break
        self._save_pending(items)

    def expire_old_tasks(self, max_age_days: int = 7) -> int:
        """Mark old pending tasks as expired. Returns count expired."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        items = self._load_pending()
        expired = 0
        for item in items:
            if item.status in ("pending", "ready"):
                created = datetime.fromisoformat(item.created_at)
                if created < cutoff:
                    item.status = "expired"
                    expired += 1
        self._save_pending(items)
        if expired:
            log.info("queue.expired", count=expired)
        return expired

    def stats(self) -> dict:
        items = self._load_pending()
        by_status: dict[str, int] = {}
        for item in items:
            by_status[item.status] = by_status.get(item.status, 0) + 1
        return {
            "pending": by_status.get("pending", 0),
            "ready": by_status.get("ready", 0),
            "failed": by_status.get("failed", 0),
            "expired": by_status.get("expired", 0),
            "completed_total": len(self._load_completed()),
        }
