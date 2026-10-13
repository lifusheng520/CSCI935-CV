"""Member5 AMP telemetry policy. No torch, RNG, training or process-kill calls.

Production decisions run synchronously INSIDE the authorized observer. A read-only
JSONL follower exists for replay/inspection; it cannot stop any PID. Rate decisions
use the complete poll snapshot, never a historical prefix. Fatal evidence latches.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

VERSION = "member5-amp-monitor-v2"
POLICY = {"consecutive_skips": 3, "skip_fraction_greater_than": 0.05,
          "rate_warmup": "one_complete_epoch", "rate_scope": "whole_current_attempt"}


class MonitorError(ValueError):
    pass


def check(ok, message):
    if not ok:
        raise MonitorError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class AMPMonitor:
    def __init__(self, attempt_id, batches_per_epoch, seeds=(0, 1, 2, 3, 4),
                 head_epochs=3, max_epochs=30, legacy_read_only=False):
        check(isinstance(attempt_id, str) and attempt_id.startswith("attempt_"), "Invalid attempt identity")
        check(type(batches_per_epoch) is int and batches_per_epoch > 0, "Invalid epoch batch count")
        self.attempt_id = attempt_id
        self.batches_per_epoch = batches_per_epoch
        self.seeds = tuple(seeds)
        self.head_epochs = head_epochs
        self.max_epochs = max_epochs
        self.legacy_read_only = legacy_read_only
        self.events = self.batches = self.skips = self.streak = self.longest_streak = 0
        self.epoch_complete = False
        self.last = None
        self.last_time = None
        self.ended = False
        self.fatal = None
        self.last_event_time = None

    def _accept(self, event):
        check(isinstance(event, dict), "Event must be an object")
        if not self.legacy_read_only:
            check(event.get("attempt_id") == self.attempt_id, "Foreign/missing attempt identity")
            check(type(event.get("event_seq")) is int and event["event_seq"] == self.events + 1,
                  "Repeated/missing/out-of-order event sequence")
        elif "attempt_id" in event:
            check(event["attempt_id"] == self.attempt_id, "Foreign attempt in legacy replay")
        check(not self.ended, "Event after terminal event")
        timestamp = datetime.fromisoformat(event["time"].replace("Z", "+00:00"))
        check(timestamp.tzinfo is not None, "Timezone-aware event time required")
        check(self.last_time is None or timestamp >= self.last_time, "Event time reversed")
        self.last_time = timestamp
        self.last_event_time = event["time"]
        self.events += 1
        if event.get("kind") == "end":
            check(type(event.get("success")) is bool, "Invalid terminal event")
            self.ended = True
            if not event["success"]:
                self.fatal = self.fatal or "Training exception/unsuccessful terminal event"
            else:
                check(self.last is not None and self.last[0] == self.seeds[-1] and
                      self.last[1] == 2 and self.last[3] == self.batches_per_epoch,
                      "Successful end before all seeds/complete epoch")
            return
        check(event.get("kind") == "batch", "Unknown telemetry kind")
        keys = ("seed", "stage", "epoch", "batch")
        check(all(type(event.get(k)) is int for k in keys), "Integer batch identity required")
        current = tuple(event[k] for k in keys)
        seed, stage, epoch, batch = current
        check(seed in self.seeds and stage in (1, 2) and 1 <= batch <= self.batches_per_epoch,
              "Batch identity outside approved range")
        check(1 <= epoch <= (self.head_epochs if stage == 1 else self.max_epochs), "Epoch outside approved range")
        first_stage = 1 if self.head_epochs else 2
        if self.last is None:
            check(current == (self.seeds[0], first_stage, 1, 1), "Missing first batch")
        else:
            s, st, ep, ba = self.last
            if seed != s:
                check(self.seeds.index(seed) == self.seeds.index(s) + 1 and st == 2 and
                      ba == self.batches_per_epoch and current[1:] == (first_stage, 1, 1),
                      "Invalid seed transition")
            elif stage != st:
                check(st == 1 and stage == 2 and ep == self.head_epochs and
                      ba == self.batches_per_epoch and (epoch, batch) == (1, 1), "Invalid stage transition")
            elif epoch != ep:
                check(epoch == ep + 1 and ba == self.batches_per_epoch and batch == 1, "Invalid epoch transition")
            else:
                check(batch == ba + 1, "Repeated/missing/out-of-order batch")
        for k in ("skipped", "loss_finite", "applied_gradients_finite", "weights_finite", "unscaled_gradients_finite"):
            check(type(event.get(k)) is bool, "Boolean telemetry required: " + k)
        values = event.get("loss_values")
        check(isinstance(values, list) and values, "Loss telemetry missing")
        loss_numbers_finite = all(type(x) in (int, float) and math.isfinite(x) for x in values)
        for k in ("scale_before", "scale_after"):
            check(type(event.get(k)) in (int, float) and math.isfinite(event[k]) and event[k] > 0,
                  "Invalid GradScaler scale")
        self.last = current
        self.batches += 1
        self.skips += int(event["skipped"])
        self.streak = self.streak + 1 if event["skipped"] else 0
        self.longest_streak = max(self.longest_streak, self.streak)
        self.epoch_complete |= batch == self.batches_per_epoch
        if not event["loss_finite"] or not loss_numbers_finite:
            self.fatal = self.fatal or "Nonfinite loss"
        if not event["applied_gradients_finite"] or (not event["skipped"] and not event["unscaled_gradients_finite"]):
            self.fatal = self.fatal or "Nonfinite optimizer-applied gradient"
        if not event["weights_finite"]:
            self.fatal = self.fatal or "Nonfinite model weight/buffer"
        if self.streak >= 3:
            self.fatal = self.fatal or "Three consecutive AMP skips"

    def consume(self, events, mode="live", pending_bytes=0):
        check(mode in ("live", "historical_snapshot", "incremental"), "Unknown observation mode")
        prior = self.batches
        try:
            for event in events:
                self._accept(event)
        except (MonitorError, KeyError, TypeError, ValueError, OverflowError) as exc:
            self.fatal = self.fatal or "Untrustworthy telemetry: " + str(exc)
        rate = self.skips / self.batches if self.batches else 0.0
        reason = self.fatal
        if reason is None and self.epoch_complete and rate > 0.05:
            reason = "Cumulative AMP skip rate exceeds 5% after complete-epoch warmup"
        if reason:
            self.fatal = reason  # A real live stop cannot be erased by later good batches.
        return {"monitor_version": VERSION, "attempt_id": self.attempt_id,
                "observed_at": utc_now(), "latest_evidence_time": self.last_event_time,
                "mode": mode, "action": "STOP" if reason else "CONTINUE", "reason": reason,
                "batches": self.batches, "new_batches": self.batches - prior,
                "optimizer_updates": self.batches - self.skips, "skips": self.skips,
                "skip_fraction": rate, "consecutive_skips": self.streak,
                "max_consecutive_skips": self.longest_streak,
                "rate_gate_active": self.epoch_complete, "batches_per_epoch": self.batches_per_epoch,
                "warning": "Isolated native GradScaler skip recorded" if self.skips and not reason else None,
                "pending_bytes": pending_bytes, "training_end_observed": self.ended}


class JSONLFollower:
    """Read-only, offset-based snapshots. Never accepts a PID or launches a process."""
    def __init__(self, path, monitor):
        self.path = Path(path).absolute()
        check(self.path.name == "amp_events.jsonl" and self.path.parent.name == monitor.attempt_id,
              "Telemetry path is not bound to current attempt")
        self.monitor = monitor
        self.offset = 0
        self.pending = b""
        self.identity = None
        self.polls = 0

    def poll(self, final=False):
        mode = "historical_snapshot" if self.polls == 0 else "incremental"
        self.polls += 1
        try:
            check(not self.path.is_symlink(), "Telemetry symlink rejected")
            with self.path.open("rb") as f:
                st = os.fstat(f.fileno())
                check(not (getattr(st, "st_file_attributes", 0) & 0x400), "Telemetry reparse point rejected")
                identity = (st.st_dev, st.st_ino)
                check(self.identity is None or identity == self.identity, "Telemetry file replaced")
                check(st.st_size >= self.offset, "Telemetry file truncated")
                self.identity = identity
                f.seek(self.offset)
                data = f.read(st.st_size - self.offset)  # snapshot boundary, not arbitrary line prefixes
                self.offset += len(data)
            pieces = (self.pending + data).split(b"\n")
            self.pending = pieces.pop()
            check(len(self.pending) <= 1024 * 1024, "Oversized incomplete JSON line")
            events = []
            def no_constant(value):
                raise MonitorError("Nonfinite JSON constant: " + value)
            def unique(pairs):
                obj = {}
                for k, v in pairs:
                    check(k not in obj, "Duplicate JSON key")
                    obj[k] = v
                return obj
            for line in pieces:
                check(bool(line.strip()), "Empty telemetry line")
                events.append(json.loads(line.decode("utf-8"), parse_constant=no_constant, object_pairs_hook=unique))
            check(not final or not self.pending, "Incomplete terminal JSONL line")
            return self.monitor.consume(events, mode, len(self.pending))
        except (OSError, ValueError, TypeError) as exc:
            self.monitor.fatal = "Untrustworthy telemetry: " + str(exc)
            return self.monitor.consume([], mode, len(self.pending))
