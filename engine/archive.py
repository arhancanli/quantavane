"""Every attempt, kept or not, goes in one SQLite file per problem. The archive is the public record of the search."""
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY,
  parent_id INTEGER,
  island INTEGER NOT NULL,
  program TEXT NOT NULL,
  valid INTEGER NOT NULL,
  score REAL,
  note TEXT,
  output TEXT,
  proposer TEXT NOT NULL,
  cost_usd REAL NOT NULL DEFAULT 0,
  seconds REAL,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS attempts_island_score ON attempts (island, valid, score);
"""


class Archive:
    def __init__(self, path, direction="max"):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.order = "DESC" if direction == "max" else "ASC"

    def add(self, *, island, program, verdict, output, proposer, parent_id=None, cost_usd=0.0, seconds=None):
        cur = self.db.execute(
            "INSERT INTO attempts (parent_id, island, program, valid, score, note, output, proposer, cost_usd, seconds, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (parent_id, island, program, int(verdict.valid), verdict.score if verdict.valid else None, verdict.note,
             json.dumps(output) if verdict.valid else None, proposer, cost_usd, seconds, time.time()))
        return cur.lastrowid

    def top(self, island=None, k=5):
        where = "valid = 1" + ("" if island is None else f" AND island = {int(island)}")
        return self.db.execute(f"SELECT * FROM attempts WHERE {where} ORDER BY score {self.order}, id LIMIT ?", (k,)).fetchall()

    def best(self):
        rows = self.top(k=1)
        return rows[0] if rows else None

    def best_solution(self):
        # Early archives stored migrated copies without their output; skip those.
        return self.db.execute("SELECT * FROM attempts WHERE valid = 1 AND output IS NOT NULL AND output != 'null'"
                               f" ORDER BY score {self.order}, id LIMIT 1").fetchone()

    def recent_failures(self, island, k=2):
        return self.db.execute("SELECT note FROM attempts WHERE island = ? AND valid = 0 ORDER BY id DESC LIMIT ?", (island, k)).fetchall()

    def stats(self):
        agg = "MAX" if self.order == "DESC" else "MIN"
        r = self.db.execute(f"SELECT COUNT(*) n, SUM(valid) ok, SUM(cost_usd) cost, {agg}(score) best, MAX(created_at) last"
                            " FROM attempts").fetchone()
        return dict(r)
