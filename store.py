"""Persistence layer.

Two interchangeable backends behind one tiny document-store interface:

* ``SqliteStore``   - zero-setup local backend (default; used for tests and local demos).
* ``DynamoStore``   - Amazon DynamoDB backend (set ``TRIPWIRE_STORE=dynamodb``).

Every decision (with its tripwires and history) is one small JSON document keyed by
(household, kind, key), so a decision made in one Alexa conversation can be checked
weeks later in another.

Concurrency: the MCP server is stateless and may run as several containers, so two requests can
touch the same decision at once. Every document carries a version number and every write can be
conditional on it (compare-and-swap). ``update()`` wraps the read-modify-write loop and retries on
conflict, so concurrent edits are merged instead of one silently overwriting the other.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable


class ConflictError(Exception):
    """A conditional write lost a race (someone else changed the document first)."""


class Store(ABC):
    @abstractmethod
    def get_v(self, household: str, kind: str, key: str) -> tuple[dict[str, Any] | None, int]:
        """Return (document or None, version). A missing document has version 0."""

    @abstractmethod
    def put(self, household: str, kind: str, key: str, data: dict[str, Any], expected_ver: int | None = None) -> int:
        """Write a document; if ``expected_ver`` is given the write only succeeds when the stored
        version matches (0 means 'must not exist'). Raises ConflictError otherwise. Returns the new version."""

    @abstractmethod
    def delete(self, household: str, kind: str, key: str) -> None: ...

    @abstractmethod
    def list(self, household: str, kind: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def purge(self, household: str) -> int:
        """Delete everything stored for a household (demos, tests, data-deletion requests)."""

    @abstractmethod
    def ping(self) -> None:
        """Cheap round-trip to the backing store; raises if it is unreachable."""

    def get(self, household: str, kind: str, key: str) -> dict[str, Any] | None:
        return self.get_v(household, kind, key)[0]

    def update(self, household: str, kind: str, key: str,
               fn: Callable[[dict[str, Any] | None], dict[str, Any] | None], retries: int = 8) -> dict[str, Any] | None:
        """Atomic read-modify-write. ``fn`` receives a private copy of the current document (or None) and
        returns the new document, or None for 'no change'. It may run more than once, so it must be a pure
        function of its argument (collect side results in a local that it overwrites on each call)."""
        for _ in range(retries):
            doc, ver = self.get_v(household, kind, key)
            new = fn(copy.deepcopy(doc) if doc is not None else None)
            if new is None:
                return None
            try:
                self.put(household, kind, key, new, expected_ver=ver)
                return new
            except ConflictError:
                continue
        raise ConflictError(f"could not update {kind}#{key} after {retries} attempts")


class SqliteStore(Store):
    def __init__(self, path: str = "tripwire.db") -> None:
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "household TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL, "
            "data TEXT NOT NULL, ver INTEGER NOT NULL DEFAULT 1, PRIMARY KEY (household, kind, key))"
        )
        self._db.commit()

    def get_v(self, household, kind, key):
        with self._lock:
            row = self._db.execute(
                "SELECT data, ver FROM docs WHERE household=? AND kind=? AND key=?", (household, kind, key)
            ).fetchone()
        return (json.loads(row[0]), row[1]) if row else (None, 0)

    def put(self, household, kind, key, data, expected_ver=None):
        with self._lock:  # one process-wide lock makes check-then-write atomic for this backend
            row = self._db.execute(
                "SELECT ver FROM docs WHERE household=? AND kind=? AND key=?", (household, kind, key)
            ).fetchone()
            cur = row[0] if row else 0
            if expected_ver is not None and cur != expected_ver:
                raise ConflictError(f"{kind}#{key}: expected v{expected_ver}, found v{cur}")
            self._db.execute(
                "INSERT INTO docs(household, kind, key, data, ver) VALUES (?,?,?,?,?) "
                "ON CONFLICT(household, kind, key) DO UPDATE SET data=excluded.data, ver=excluded.ver",
                (household, kind, key, json.dumps(data), cur + 1),
            )
            self._db.commit()
            return cur + 1

    def delete(self, household, kind, key):
        with self._lock:
            self._db.execute("DELETE FROM docs WHERE household=? AND kind=? AND key=?", (household, kind, key))
            self._db.commit()

    def list(self, household, kind):
        with self._lock:
            rows = self._db.execute(
                "SELECT key, data FROM docs WHERE household=? AND kind=? ORDER BY key", (household, kind)
            ).fetchall()
        return [{**json.loads(d), "_key": k} for k, d in rows]

    def purge(self, household):
        with self._lock:
            n = self._db.execute("DELETE FROM docs WHERE household=?", (household,)).rowcount
            self._db.commit()
        return n

    def ping(self):
        with self._lock:
            self._db.execute("SELECT 1").fetchone()


class DynamoStore(Store):
    """Single-table DynamoDB design: pk = HOUSEHOLD#<id>, sk = <kind>#<key>; attributes ``data`` (JSON) and ``ver``."""

    def __init__(self, table_name: str, region: str | None = None, client: Any = None) -> None:
        if client is None:
            import boto3  # imported lazily so local mode needs no AWS config
            from botocore.config import Config

            # Bounded latency: a voice request must not hang on a sick network. Retries cover throttling/5xx.
            cfg = Config(
                connect_timeout=float(os.environ.get("TRIPWIRE_DDB_CONNECT_TIMEOUT", "2")),
                read_timeout=float(os.environ.get("TRIPWIRE_DDB_READ_TIMEOUT", "4")),
                retries={"total_max_attempts": int(os.environ.get("TRIPWIRE_DDB_ATTEMPTS", "3")), "mode": "standard"},
            )
            client = boto3.client("dynamodb", region_name=region, config=cfg)
        self._ddb = client
        self._table = table_name

    @staticmethod
    def _pk(household: str) -> dict[str, str]:
        return {"S": f"HOUSEHOLD#{household}"}

    def _key(self, household, kind, key):
        return {"pk": self._pk(household), "sk": {"S": f"{kind}#{key}"}}

    def get_v(self, household, kind, key):
        item = self._ddb.get_item(TableName=self._table, Key=self._key(household, kind, key), ConsistentRead=True).get("Item")
        if not item:
            return None, 0
        return json.loads(item["data"]["S"]), int(item.get("ver", {"N": "1"})["N"])

    def put(self, household, kind, key, data, expected_ver=None):
        kwargs: dict[str, Any] = dict(
            TableName=self._table, Key=self._key(household, kind, key),
            UpdateExpression="SET #d = :d ADD #v :one",
            ExpressionAttributeNames={"#d": "data", "#v": "ver"},
            ExpressionAttributeValues={":d": {"S": json.dumps(data)}, ":one": {"N": "1"}},
            ReturnValues="UPDATED_NEW",
        )
        if expected_ver is not None:
            if expected_ver == 0:
                kwargs["ConditionExpression"] = "attribute_not_exists(pk)"
            else:
                kwargs["ConditionExpression"] = "#v = :e"
                kwargs["ExpressionAttributeValues"][":e"] = {"N": str(expected_ver)}
        try:
            resp = self._ddb.update_item(**kwargs)
        except self._ddb.exceptions.ConditionalCheckFailedException as e:
            raise ConflictError(f"{kind}#{key}: version mismatch") from e
        return int(resp["Attributes"]["ver"]["N"])

    def delete(self, household, kind, key):
        self._ddb.delete_item(TableName=self._table, Key=self._key(household, kind, key))

    def _query(self, household: str, prefix: str | None):
        kwargs: dict[str, Any] = {"TableName": self._table, "ConsistentRead": True,
                                  "KeyConditionExpression": "pk = :pk", "ExpressionAttributeValues": {":pk": self._pk(household)}}
        if prefix:
            kwargs["KeyConditionExpression"] += " AND begins_with(sk, :p)"
            kwargs["ExpressionAttributeValues"][":p"] = {"S": prefix}
        while True:
            resp = self._ddb.query(**kwargs)
            yield from resp.get("Items", [])
            if "LastEvaluatedKey" not in resp:
                return
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def list(self, household, kind):
        out = [{**json.loads(i["data"]["S"]), "_key": i["sk"]["S"].split("#", 1)[1]} for i in self._query(household, f"{kind}#")]
        return sorted(out, key=lambda d: d["_key"])

    def purge(self, household):
        n = 0
        for i in list(self._query(household, None)):
            self._ddb.delete_item(TableName=self._table, Key={"pk": i["pk"], "sk": i["sk"]})
            n += 1
        return n

    def ping(self):
        self._ddb.get_item(TableName=self._table, Key=self._key("_health", "ping", "x"))


def make_store() -> Store:
    backend = os.environ.get("TRIPWIRE_STORE", "sqlite").lower()
    if backend == "dynamodb":
        return DynamoStore(os.environ.get("TRIPWIRE_TABLE", "Tripwire"), region=os.environ.get("AWS_REGION"))
    return SqliteStore(os.environ.get("TRIPWIRE_DB", "tripwire.db"))
