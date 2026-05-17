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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pcbs (
                    board_number TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    ll_x_mm REAL NOT NULL,
                    ll_y_mm REAL NOT NULL,
                    relative_z_mm REAL NOT NULL,
                    rotation_deg REAL NOT NULL,
                    items_json TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS panels (
                    panel_name TEXT PRIMARY KEY,
                    source_board_number TEXT NOT NULL,
                    count_x INTEGER NOT NULL,
                    count_y INTEGER NOT NULL,
                    pitch_x_mm REAL NOT NULL,
                    pitch_y_mm REAL NOT NULL,
                    rotation_deg REAL NOT NULL,
                    import_type TEXT NOT NULL DEFAULT '',
                    import_file TEXT NOT NULL DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_name TEXT PRIMARY KEY,
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
            "pcbs": self._table_count("pcbs"),
            "panels": self._table_count("panels"),
            "jobs": self._table_count("jobs"),
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

    def load_pcbs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT board_number, name, version, ll_x_mm, ll_y_mm, relative_z_mm, rotation_deg, items_json
                FROM pcbs
                ORDER BY board_number
                """
            ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                items_raw = json.loads(str(row["items_json"]))
                items = items_raw if isinstance(items_raw, list) else []
            except Exception:
                items = []
            out.append(
                {
                    "board_number": str(row["board_number"]).strip().upper(),
                    "name": str(row["name"]),
                    "version": str(row["version"]),
                    "ll_x_mm": float(row["ll_x_mm"]),
                    "ll_y_mm": float(row["ll_y_mm"]),
                    "relative_z_mm": float(row["relative_z_mm"]),
                    "rotation_deg": float(row["rotation_deg"]),
                    "items": items,
                }
            )
        return out

    def upsert_pcb(self, pcb: dict[str, Any]) -> None:
        board_number = str(pcb.get("board_number", "")).strip().upper()
        if not board_number:
            return
        items = pcb.get("items", [])
        if not isinstance(items, list):
            items = []
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO pcbs(board_number, name, version, ll_x_mm, ll_y_mm, relative_z_mm, rotation_deg, items_json)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(board_number) DO UPDATE SET
                    name=excluded.name,
                    version=excluded.version,
                    ll_x_mm=excluded.ll_x_mm,
                    ll_y_mm=excluded.ll_y_mm,
                    relative_z_mm=excluded.relative_z_mm,
                    rotation_deg=excluded.rotation_deg,
                    items_json=excluded.items_json
                """,
                (
                    board_number,
                    str(pcb.get("name", "")).strip(),
                    str(pcb.get("version", "")).strip(),
                    float(pcb.get("ll_x_mm", 0.0) or 0.0),
                    float(pcb.get("ll_y_mm", 0.0) or 0.0),
                    float(pcb.get("relative_z_mm", 0.0) or 0.0),
                    float(pcb.get("rotation_deg", 0.0) or 0.0),
                    json.dumps(items),
                ),
            )
            conn.commit()

    def delete_pcb(self, board_number: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM pcbs WHERE board_number = ?", (str(board_number).strip().upper(),))
            conn.commit()

    def load_panels(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT panel_name, source_board_number, count_x, count_y, pitch_x_mm, pitch_y_mm, rotation_deg, import_type, import_file
                FROM panels
                ORDER BY panel_name
                """
            ).fetchall()
        return [
            {
                "panel_name": str(row["panel_name"]).strip().upper(),
                "source_board_number": str(row["source_board_number"]).strip().upper(),
                "count_x": int(row["count_x"]),
                "count_y": int(row["count_y"]),
                "pitch_x_mm": float(row["pitch_x_mm"]),
                "pitch_y_mm": float(row["pitch_y_mm"]),
                "rotation_deg": float(row["rotation_deg"]),
                "import_type": str(row["import_type"]),
                "import_file": str(row["import_file"]),
            }
            for row in rows
        ]

    def upsert_panel(self, panel: dict[str, Any]) -> None:
        panel_name = str(panel.get("panel_name", "")).strip().upper()
        if not panel_name:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO panels(panel_name, source_board_number, count_x, count_y, pitch_x_mm, pitch_y_mm, rotation_deg, import_type, import_file)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(panel_name) DO UPDATE SET
                    source_board_number=excluded.source_board_number,
                    count_x=excluded.count_x,
                    count_y=excluded.count_y,
                    pitch_x_mm=excluded.pitch_x_mm,
                    pitch_y_mm=excluded.pitch_y_mm,
                    rotation_deg=excluded.rotation_deg,
                    import_type=excluded.import_type,
                    import_file=excluded.import_file
                """,
                (
                    panel_name,
                    str(panel.get("source_board_number", "")).strip().upper(),
                    int(panel.get("count_x", 1) or 1),
                    int(panel.get("count_y", 1) or 1),
                    float(panel.get("pitch_x_mm", 0.0) or 0.0),
                    float(panel.get("pitch_y_mm", 0.0) or 0.0),
                    float(panel.get("rotation_deg", 0.0) or 0.0),
                    str(panel.get("import_type", "")).strip(),
                    str(panel.get("import_file", "")).strip(),
                ),
            )
            conn.commit()

    def delete_panel(self, panel_name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM panels WHERE panel_name = ?", (str(panel_name).strip().upper(),))
            conn.commit()

    def load_jobs(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_name, payload
                FROM jobs
                ORDER BY job_name
                """
            ).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(str(row["payload"]))
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload.setdefault("job_name", str(row["job_name"]).strip().upper())
            out.append(payload)
        return out

    def upsert_job(self, job: dict[str, Any]) -> None:
        job_name = str(job.get("job_name", "")).strip().upper()
        if not job_name:
            return
        payload = dict(job)
        payload["job_name"] = job_name
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(job_name, payload)
                VALUES(?, ?)
                ON CONFLICT(job_name) DO UPDATE SET
                    payload=excluded.payload
                """,
                (job_name, json.dumps(payload)),
            )
            conn.commit()

    def delete_job(self, job_name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE job_name = ?", (str(job_name).strip().upper(),))
            conn.commit()
