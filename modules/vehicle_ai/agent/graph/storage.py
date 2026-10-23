"""SQLite checkpoint 和执行日志；不确定的写入尝试不得自动重放。"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from uuid import uuid4
from pathlib import Path
from threading import Event, RLock, Thread

from langgraph.checkpoint.sqlite import SqliteSaver

from modules.vehicle_ai.tools.base import ToolResult


class GraphStorage:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            path = Path(path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.checkpointer = SqliteSaver(self.connection)
        # checkpoint 由 LangGraph 后台线程写入，执行日志必须使用独立连接。
        self.execution_connection = sqlite3.connect(str(path), check_same_thread=False)
        self.execution_connection.execute("PRAGMA busy_timeout=5000")
        self.lock = RLock()
        self.owner = uuid4().hex
        self.lease_depth = 0
        self.lease_error: Exception | None = None
        self.execution_connection.execute(
            "CREATE TABLE IF NOT EXISTS vehiclemind_executions "
            "(thread_id TEXT, operation_id TEXT, signature TEXT NOT NULL, "
            "status TEXT NOT NULL, payload TEXT, PRIMARY KEY(thread_id, operation_id))"
        )
        self.execution_connection.execute(
            "CREATE TABLE IF NOT EXISTS vehiclemind_sessions "
            "(thread_id TEXT PRIMARY KEY, owner TEXT NOT NULL, expires_at REAL NOT NULL)"
        )
        self.execution_connection.commit()

    @contextmanager
    def session(self, thread_id: str, seconds: float):
        """同一会话只允许一个执行者；进程退出后租约到期才可恢复。"""
        heartbeat_stop = None
        heartbeat = None
        with self.lock:
            if self.lease_depth:
                self.lease_depth += 1
            else:
                self.execution_connection.execute("BEGIN IMMEDIATE")
                row = self.execution_connection.execute(
                    "SELECT owner, expires_at FROM vehiclemind_sessions WHERE thread_id=?",
                    (thread_id,),
                ).fetchone()
                if row and row[0] != self.owner and row[1] > time.time():
                    self.execution_connection.rollback()
                    raise RuntimeError("SESSION_BUSY: another worker owns this thread")
                self.execution_connection.execute(
                    "INSERT OR REPLACE INTO vehiclemind_sessions VALUES (?, ?, ?)",
                    (thread_id, self.owner, time.time() + seconds),
                )
                self.execution_connection.commit()
                self.lease_depth = 1
                self.lease_error = None
                # 节点内的外部调用可能阻塞，租约必须独立于节点调度持续更新。
                heartbeat_stop = Event()
                heartbeat = Thread(
                    target=self._heartbeat,
                    args=(thread_id, seconds, heartbeat_stop),
                    daemon=True,
                )
                heartbeat.start()
        try:
            yield
        finally:
            if heartbeat_stop is not None and heartbeat is not None:
                heartbeat_stop.set()
                heartbeat.join()
            with self.lock:
                self.lease_depth -= 1
                if not self.lease_depth:
                    self.execution_connection.execute(
                        "DELETE FROM vehiclemind_sessions WHERE thread_id=? AND owner=?",
                        (thread_id, self.owner),
                    )
                    self.execution_connection.commit()

    def _heartbeat(self, thread_id: str, seconds: float, stop: Event) -> None:
        while not stop.wait(max(0.001, min(10.0, seconds / 3))):
            try:
                self.renew_session(thread_id, seconds)
            except Exception as error:
                self.lease_error = error
                return

    def renew_session(self, thread_id: str, seconds: float) -> None:
        if not self.lease_depth:
            return
        with self.lock:
            if self.lease_error is not None:
                raise RuntimeError("SESSION_LEASE_LOST") from self.lease_error
            cursor = self.execution_connection.execute(
                "UPDATE vehiclemind_sessions SET expires_at=? WHERE thread_id=? AND owner=?",
                (time.time() + seconds, thread_id, self.owner),
            )
            self.execution_connection.commit()
            if cursor.rowcount != 1:
                raise RuntimeError("SESSION_LEASE_LOST")

    def lookup(self, thread_id: str, operation_id: str, signature: str):
        row = self.execution_connection.execute(
            "SELECT signature, status, payload FROM vehiclemind_executions "
            "WHERE thread_id=? AND operation_id=?",
            (thread_id, operation_id),
        ).fetchone()
        if row is None:
            return None
        return self._stored_payload(row, signature)

    @staticmethod
    def _stored_payload(row, signature):
        if row[0] != signature:
            return {
                "result": ToolResult(
                    False,
                    "执行标识绑定的参数不一致，操作已停止。",
                    error="EXECUTION_MISMATCH",
                ).to_dict()
            }
        if row[1] == "DONE":
            return json.loads(row[2])
        return {
            "result": ToolResult(
                False,
                "上次写入结果尚不确定，禁止自动重放。",
                error="WRITE_OUTCOME_UNKNOWN",
                data={"outcome_unknown": True},
            ).to_dict()
        }

    def run_once(self, thread_id: str, operation_id: str, signature: str, execute):
        """先提交 STARTED；中途崩溃后返回未知结果，不再次调用写入 handler。"""
        with self.lock:
            self.execution_connection.execute("BEGIN IMMEDIATE")
            row = self.execution_connection.execute(
                "SELECT signature, status, payload FROM vehiclemind_executions "
                "WHERE thread_id=? AND operation_id=?",
                (thread_id, operation_id),
            ).fetchone()
            if row is not None:
                self.execution_connection.commit()
                return self._stored_payload(row, signature)
            self.execution_connection.execute(
                "INSERT INTO vehiclemind_executions VALUES (?, ?, ?, 'STARTED', NULL)",
                (thread_id, operation_id, signature),
            )
            self.execution_connection.commit()
        # 外部 handler 不持有数据库连接锁，使心跳可在阻塞期间续租。
        payload = execute()
        with self.lock:
            self.execution_connection.execute(
                "UPDATE vehiclemind_executions SET status='DONE', payload=? "
                "WHERE thread_id=? AND operation_id=?",
                (
                    json.dumps(payload, ensure_ascii=False, default=str),
                    thread_id,
                    operation_id,
                ),
            )
            self.execution_connection.commit()
            return payload

    def latest_vehicle(self, thread_id: str) -> dict | None:
        """成功落盘的模拟后端状态优先于尚未更新的 graph checkpoint。"""
        rows = self.execution_connection.execute(
            "SELECT payload FROM vehiclemind_executions "
            "WHERE thread_id=? AND status='DONE' ORDER BY rowid DESC",
            (thread_id,),
        )
        for row in rows:
            vehicle = json.loads(row[0]).get("vehicle")
            if vehicle:
                return vehicle
        return None

    def close(self):
        self.connection.close()
        self.execution_connection.close()
