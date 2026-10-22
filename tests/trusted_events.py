"""Trusted pytest lifecycle recording, separate from candidate execution."""

from pathlib import Path
import json, os

ROOT = Path("/logs/verifier")
record = {}


def pytest_sessionstart(session):
    global record
    identity = json.loads((ROOT / "run-identity.json").read_text())
    record = {
        "run_id": identity["run_id"],
        "collected": [],
        "reports": {},
        "session_finished": False,
    }


def pytest_collection_finish(session):
    record["collected"] = [item.nodeid for item in session.items]


def pytest_runtest_logreport(report):
    record["reports"].setdefault(report.nodeid, {})[report.when] = {
        "outcome": report.outcome,
        "duration": report.duration,
    }


def pytest_sessionfinish(session, exitstatus):
    record["session_finished"] = True
    record["exitstatus"] = int(exitstatus)
    target = ROOT / "execution.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
