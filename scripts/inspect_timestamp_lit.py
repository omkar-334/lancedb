"""Diagnose how lit(datetime) renders against tz-aware vs naive columns.

Prints:
  1. The SQL emitted by lit() for naive vs aware datetimes.
  2. The actual rows returned by `<` filters for the five scenarios in the
     task brief, against a real LanceDB table.

Useful for confirming or contradicting the claim that the lit() implementation
in #3235 always strips tzinfo.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date, datetime, timedelta, timezone

import pyarrow as pa

import lancedb
from lancedb.expr import col, lit


def header(s: str) -> None:
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def show_sql(label: str, value) -> None:
    expr = lit(value)
    sql = expr.to_sql()
    print(f"  {label:<55s} → {sql}")


def make_table(uri: str, name: str, column_tz):
    db = lancedb.connect(uri)
    schema = pa.schema([("ts", pa.timestamp("us", tz=column_tz))])
    if column_tz is None:
        rows = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    else:
        rows = [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        ]
    return db.create_table(
        name, pa.table({"ts": rows}, schema=schema), mode="overwrite"
    )


def run_filter(table, cutoff, label: str) -> dict:
    expr = col("ts") < lit(cutoff)
    sql = expr.to_sql()
    out = {"label": label, "cutoff": repr(cutoff), "sql": sql}
    try:
        result = table.search().where(expr).to_arrow()
        out["num_rows"] = result.num_rows
        out["rows"] = [r.as_py().isoformat() for r in result["ts"]]
        out["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
    print(json.dumps(out, default=str, indent=2))
    return out


def main() -> None:
    header("SQL emitted by lit() for various inputs")
    show_sql("date(2024, 1, 1)", date(2024, 1, 1))
    show_sql("naive datetime 2024-01-01T00:00:00", datetime(2024, 1, 1, 0, 0, 0))
    show_sql("naive datetime 2024-01-01T12:00:00", datetime(2024, 1, 1, 12, 0, 0))
    show_sql(
        "datetime 2024-01-01T12:00:00 UTC",
        datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    show_sql(
        "datetime 2024-01-01T04:00:00 PST (-08:00)",
        datetime(2024, 1, 1, 4, 0, 0, tzinfo=timezone(timedelta(hours=-8))),
    )

    tmp = tempfile.mkdtemp(prefix="lancedb_ts_")
    try:
        results = []

        header("1) Both naive (column tz=None, literal naive)")
        t = make_table(tmp, "ts_naive", column_tz=None)
        results.append(run_filter(t, datetime(2024, 1, 1, 12, 0, 0), "both_naive"))

        header("2) Both same tz (column tz=UTC, literal tz=UTC)")
        t = make_table(tmp, "ts_utc", column_tz="UTC")
        results.append(
            run_filter(
                t,
                datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "both_utc",
            )
        )

        header("3) Different tz (column tz=UTC, literal tz=PST)")
        t = make_table(tmp, "ts_utc2", column_tz="UTC")
        pst = timezone(timedelta(hours=-8))
        results.append(
            run_filter(
                t,
                datetime(2024, 1, 1, 4, 0, 0, tzinfo=pst),
                "different_tz",
            )
        )

        header("4) Tz column, naive literal (column tz=UTC, literal naive)")
        t = make_table(tmp, "ts_utc3", column_tz="UTC")
        results.append(
            run_filter(t, datetime(2024, 1, 1, 12, 0, 0), "tz_col_naive_lit")
        )

        header("5) Naive column, tz literal (column tz=None, literal tz=UTC)")
        t = make_table(tmp, "ts_naive2", column_tz=None)
        results.append(
            run_filter(
                t,
                datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                "naive_col_tz_lit",
            )
        )

        header("Summary")
        for r in results:
            status = r["status"]
            if status == "ok":
                print(f"  {r['label']:<22s} rows={r['num_rows']}  sql={r['sql']}")
            else:
                print(f"  {r['label']:<22s} ERROR  sql={r['sql']}  {r['error']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
