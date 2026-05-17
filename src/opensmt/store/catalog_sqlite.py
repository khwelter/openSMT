from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class CatalogSQLite:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS packages (
                    name TEXT PRIMARY KEY,
                    footprint TEXT NOT NULL,
                    length_mm REAL NOT NULL,
                    width_mm REAL NOT NULL,
                    height_mm REAL NOT NULL,
                    pin_count INTEGER NOT NULL,
                    compatible_nozzle_tips TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parts (
                    part_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    package TEXT NOT NULL,
                    quantity INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feeders (
                    feeder_id TEXT PRIMARY KEY,
                    feeder_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _table_count(self, table: str) -> int:
        with self._connect() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
            return int(row["c"]) if row is not None else 0

    def counts(self) -> dict[str, int]:
        return {
            "packages": self._table_count("packages"),
            "parts": self._table_count("parts"),
            "feeders": self._table_count("feeders"),
        }

    def bootstrap_packages_from_dir(self, config_dir: str | Path) -> None:
        if self._table_count("packages") > 0:
            return
        root = Path(config_dir).expanduser()
        if not root.exists() or not root.is_dir():
            return
        for path in sorted(root.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            self.upsert_package(raw)

    def bootstrap_parts_from_file(self, parts_file: str | Path) -> None:
        if self._table_count("parts") > 0:
            return
        path = Path(parts_file).expanduser()
        if not path.exists() or not path.is_file():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        items = raw.get("parts", []) if isinstance(raw, dict) else []
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict):
                self.upsert_part(item)

    def bootstrap_feeders(self, feeders: list[dict[str, Any]]) -> None:
        if self._table_count("feeders") > 0:
            return
        for feeder in feeders:
            if isinstance(feeder, dict):
                self.upsert_feeder(feeder)

    def load_packages(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT name, footprint, length_mm, width_mm, height_mm, pin_count, compatible_nozzle_tips
                FROM packages
                ORDER BY name
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                compat = json.loads(str(row["compatible_nozzle_tips"]))
                compat_list = [str(v).strip() for v in compat if str(v).strip()] if isinstance(compat, list) else []
            except Exception:
                compat_list = []
            out.append(
                {
                    "name": str(row["name"]).strip().upper(),
                    "footprint": str(row["footprint"]),
                    "length_mm": float(row["length_mm"]),
                    "width_mm": float(row["width_mm"]),
                    "height_mm": float(row["height_mm"]),
                    "pin_count": int(row["pin_count"]),
                    "compatible_nozzle_tips": compat_list,
                }
            )
        return out

    def upsert_package(self, package: dict[str, Any]) -> None:
        name = str(package.get("name", "")).strip().upper()
        if not name:
            return
        compat = [str(v).strip() for v in package.get("compatible_nozzle_tips", []) if str(v).strip()]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO packages(name, footprint, length_mm, width_mm, height_mm, pin_count, compatible_nozzle_tips)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    footprint=excluded.footprint,
                    length_mm=excluded.length_mm,
                    width_mm=excluded.width_mm,
                    height_mm=excluded.height_mm,
                    pin_count=excluded.pin_count,
                    compatible_nozzle_tips=excluded.compatible_nozzle_tips
                """,
                (
                    name,
                    str(package.get("footprint", "")).strip(),
                    float(package.get("length_mm", 0.0) or 0.0),
                    float(package.get("width_mm", 0.0) or 0.0),
                    float(package.get("height_mm", 0.0) or 0.0),
                    int(package.get("pin_count", 0) or 0),
                    json.dumps(compat),
                ),
            )
            conn.commit()

    def delete_package(self, name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM packages WHERE name = ?", (str(name).strip().upper(),))
            conn.commit()

    def load_parts(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT part_id, description, package, quantity
                FROM parts
                ORDER BY part_id
                """
            ).fetchall()
        return [
            {
                "part_id": str(row["part_id"]).strip().upper(),
                "description": str(row["description"]),
                "package": str(row["package"]).strip().upper(),
                "quantity": int(row["quantity"]),
            }
            for row in rows
        ]

    def upsert_part(self, part: dict[str, Any]) -> None:
        part_id = str(part.get("part_id", "")).strip().upper()
        if not part_id:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO parts(part_id, description, package, quantity)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(part_id) DO UPDATE SET
                    description=excluded.description,
                    package=excluded.package,
                    quantity=excluded.quantity
                """,
                (
                    part_id,
                    str(part.get("description", "")).strip(),
                    str(part.get("package", "")).strip().upper(),
                    int(part.get("quantity", 0) or 0),
                ),
            )
            conn.commit()

    def delete_part(self, part_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM parts WHERE part_id = ?", (str(part_id).strip().upper(),))
            conn.commit()

    def load_feeders(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload
                FROM feeders
                ORDER BY feeder_id
                """
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload"]))
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def upsert_feeder(self, feeder: dict[str, Any]) -> None:
        feeder_id = str(feeder.get("feeder_id", "")).strip().upper()
        if not feeder_id:
            return
        feeder_type = str(feeder.get("feeder_type", "")).strip().lower()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feeders(feeder_id, feeder_type, payload)
                VALUES(?, ?, ?)
                ON CONFLICT(feeder_id) DO UPDATE SET
                    feeder_type=excluded.feeder_type,
                    payload=excluded.payload
                """,
                (feeder_id, feeder_type, json.dumps(feeder)),
            )
            conn.commit()
