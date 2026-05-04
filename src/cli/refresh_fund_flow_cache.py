"""定向刷新 prepared factors cache 中的 fund_flow 特征，从 scripts/refresh_prepared_fund_flow_cache.py 迁入。"""

from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

FUND_FLOW_FEATURE_COLS: tuple[str, ...] = (
    "main_inflow_z_5d",
    "main_inflow_z_10d",
    "main_inflow_z_20d",
    "super_inflow_z_5d",
    "super_inflow_z_10d",
    "super_inflow_z_20d",
    "flow_divergence_5d",
    "flow_divergence_10d",
    "flow_divergence_20d",
    "main_inflow_streak",
)


def _sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def refresh_fund_flow_cache(
    *,
    input_cache: Path,
    output_cache: Path | None = None,
    duckdb_path: Path,
    flow_table: str = "a_share_fund_flow",
) -> None:
    """用 DuckDB 为 prepared factors parquet 定向回填资金流特征。"""
    out = output_cache or input_cache
    tmp_cache = out.with_suffix(out.suffix + ".tmp")
    input_meta = _meta_path(input_cache)
    output_meta = _meta_path(out)

    con = duckdb.connect()
    try:
        con.execute(f"ATTACH {_sql_string(str(duckdb_path))} AS market_db (READ_ONLY)")

        base_cols = [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM parquet_scan('{input_cache.as_posix()}')"
            ).fetchall()
        ]
        keep_cols = [c for c in base_cols if c not in FUND_FLOW_FEATURE_COLS]
        keep_expr = ",\n                    ".join(_sql_ident(c) for c in keep_cols)

        sql = f"""
        COPY (
            WITH base AS (
                SELECT *
                FROM parquet_scan({_sql_string(input_cache.as_posix())})
            ),
            flow_raw AS (
                SELECT
                    CAST(symbol AS VARCHAR) AS symbol,
                    CAST(trade_date AS DATE) AS trade_date,
                    CAST(main_net_inflow_pct AS DOUBLE) AS main_net_inflow_pct,
                    CAST(super_large_net_inflow_pct AS DOUBLE) AS super_large_net_inflow_pct,
                    CAST(small_net_inflow_pct AS DOUBLE) AS small_net_inflow_pct
                FROM market_db.{_sql_ident(flow_table)}
            ),
            joined AS (
                SELECT
                    base.*,
                    flow_raw.main_net_inflow_pct,
                    flow_raw.super_large_net_inflow_pct,
                    flow_raw.small_net_inflow_pct
                FROM base
                LEFT JOIN flow_raw
                  ON CAST(base.symbol AS VARCHAR) = flow_raw.symbol
                 AND CAST(base.trade_date AS DATE) = flow_raw.trade_date
            ),
            rolled AS (
                SELECT
                    *,
                    CASE
                        WHEN COUNT(main_net_inflow_pct) OVER w5 >= 3
                        THEN AVG(main_net_inflow_pct) OVER w5
                    END AS main_roll_5,
                    CASE
                        WHEN COUNT(main_net_inflow_pct) OVER w10 >= 5
                        THEN AVG(main_net_inflow_pct) OVER w10
                    END AS main_roll_10,
                    CASE
                        WHEN COUNT(main_net_inflow_pct) OVER w20 >= 10
                        THEN AVG(main_net_inflow_pct) OVER w20
                    END AS main_roll_20,
                    CASE
                        WHEN COUNT(super_large_net_inflow_pct) OVER w5 >= 3
                        THEN AVG(super_large_net_inflow_pct) OVER w5
                    END AS super_roll_5,
                    CASE
                        WHEN COUNT(super_large_net_inflow_pct) OVER w10 >= 5
                        THEN AVG(super_large_net_inflow_pct) OVER w10
                    END AS super_roll_10,
                    CASE
                        WHEN COUNT(super_large_net_inflow_pct) OVER w20 >= 10
                        THEN AVG(super_large_net_inflow_pct) OVER w20
                    END AS super_roll_20,
                    CASE
                        WHEN COUNT(small_net_inflow_pct) OVER w5 >= 3
                        THEN AVG(small_net_inflow_pct) OVER w5
                    END AS small_roll_5,
                    CASE
                        WHEN COUNT(small_net_inflow_pct) OVER w10 >= 5
                        THEN AVG(small_net_inflow_pct) OVER w10
                    END AS small_roll_10,
                    CASE
                        WHEN COUNT(small_net_inflow_pct) OVER w20 >= 10
                        THEN AVG(small_net_inflow_pct) OVER w20
                    END AS small_roll_20,
                    CASE
                        WHEN main_net_inflow_pct IS NULL THEN 0
                        WHEN main_net_inflow_pct > 0 THEN 1
                        ELSE -1
                    END AS streak_sign,
                    LAG(
                        CASE
                            WHEN main_net_inflow_pct IS NULL THEN 0
                            WHEN main_net_inflow_pct > 0 THEN 1
                            ELSE -1
                        END
                    ) OVER (
                        PARTITION BY CAST(symbol AS VARCHAR)
                        ORDER BY CAST(trade_date AS DATE)
                    ) AS prev_streak_sign
                FROM joined
                WINDOW
                    w5 AS (
                        PARTITION BY CAST(symbol AS VARCHAR)
                        ORDER BY CAST(trade_date AS DATE)
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ),
                    w10 AS (
                        PARTITION BY CAST(symbol AS VARCHAR)
                        ORDER BY CAST(trade_date AS DATE)
                        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                    ),
                    w20 AS (
                        PARTITION BY CAST(symbol AS VARCHAR)
                        ORDER BY CAST(trade_date AS DATE)
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    )
            ),
            streaked AS (
                SELECT
                    *,
                    (main_roll_5 - small_roll_5) AS div_roll_5,
                    (main_roll_10 - small_roll_10) AS div_roll_10,
                    (main_roll_20 - small_roll_20) AS div_roll_20,
                    SUM(
                        CASE
                            WHEN streak_sign = 0 THEN 1
                            WHEN prev_streak_sign IS NULL THEN 1
                            WHEN prev_streak_sign = 0 THEN 1
                            WHEN prev_streak_sign <> streak_sign THEN 1
                            ELSE 0
                        END
                    ) OVER (
                        PARTITION BY CAST(symbol AS VARCHAR)
                        ORDER BY CAST(trade_date AS DATE)
                    ) AS streak_grp
                FROM rolled
            ),
            with_streak AS (
                SELECT
                    *,
                    CASE
                        WHEN streak_sign = 0 THEN 0.0
                        WHEN streak_sign > 0 THEN CAST(
                            ROW_NUMBER() OVER (
                                PARTITION BY CAST(symbol AS VARCHAR), streak_grp
                                ORDER BY CAST(trade_date AS DATE)
                            ) AS DOUBLE
                        )
                        ELSE -CAST(
                            ROW_NUMBER() OVER (
                                PARTITION BY CAST(symbol AS VARCHAR), streak_grp
                                ORDER BY CAST(trade_date AS DATE)
                            ) AS DOUBLE
                        )
                    END AS main_inflow_streak
                FROM streaked
            ),
            zscored AS (
                SELECT
                    {keep_expr},
                    CASE
                        WHEN STDDEV_SAMP(main_roll_5) OVER d = 0 OR STDDEV_SAMP(main_roll_5) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((main_roll_5 - AVG(main_roll_5) OVER d) / (STDDEV_SAMP(main_roll_5) OVER d), 3.0), -3.0)
                    END AS main_inflow_z_5d,
                    CASE
                        WHEN STDDEV_SAMP(main_roll_10) OVER d = 0 OR STDDEV_SAMP(main_roll_10) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((main_roll_10 - AVG(main_roll_10) OVER d) / (STDDEV_SAMP(main_roll_10) OVER d), 3.0), -3.0)
                    END AS main_inflow_z_10d,
                    CASE
                        WHEN STDDEV_SAMP(main_roll_20) OVER d = 0 OR STDDEV_SAMP(main_roll_20) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((main_roll_20 - AVG(main_roll_20) OVER d) / (STDDEV_SAMP(main_roll_20) OVER d), 3.0), -3.0)
                    END AS main_inflow_z_20d,
                    CASE
                        WHEN STDDEV_SAMP(super_roll_5) OVER d = 0 OR STDDEV_SAMP(super_roll_5) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((super_roll_5 - AVG(super_roll_5) OVER d) / (STDDEV_SAMP(super_roll_5) OVER d), 3.0), -3.0)
                    END AS super_inflow_z_5d,
                    CASE
                        WHEN STDDEV_SAMP(super_roll_10) OVER d = 0 OR STDDEV_SAMP(super_roll_10) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((super_roll_10 - AVG(super_roll_10) OVER d) / (STDDEV_SAMP(super_roll_10) OVER d), 3.0), -3.0)
                    END AS super_inflow_z_10d,
                    CASE
                        WHEN STDDEV_SAMP(super_roll_20) OVER d = 0 OR STDDEV_SAMP(super_roll_20) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((super_roll_20 - AVG(super_roll_20) OVER d) / (STDDEV_SAMP(super_roll_20) OVER d), 3.0), -3.0)
                    END AS super_inflow_z_20d,
                    CASE
                        WHEN STDDEV_SAMP(div_roll_5) OVER d = 0 OR STDDEV_SAMP(div_roll_5) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((div_roll_5 - AVG(div_roll_5) OVER d) / (STDDEV_SAMP(div_roll_5) OVER d), 3.0), -3.0)
                    END AS flow_divergence_5d,
                    CASE
                        WHEN STDDEV_SAMP(div_roll_10) OVER d = 0 OR STDDEV_SAMP(div_roll_10) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((div_roll_10 - AVG(div_roll_10) OVER d) / (STDDEV_SAMP(div_roll_10) OVER d), 3.0), -3.0)
                    END AS flow_divergence_10d,
                    CASE
                        WHEN STDDEV_SAMP(div_roll_20) OVER d = 0 OR STDDEV_SAMP(div_roll_20) OVER d IS NULL THEN NULL
                        ELSE GREATEST(LEAST((div_roll_20 - AVG(div_roll_20) OVER d) / (STDDEV_SAMP(div_roll_20) OVER d), 3.0), -3.0)
                    END AS flow_divergence_20d,
                    main_inflow_streak
                FROM with_streak
                WINDOW d AS (PARTITION BY CAST(trade_date AS DATE))
            )
            SELECT * FROM zscored
        ) TO {_sql_string(tmp_cache.as_posix())} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(sql)
        tmp_cache.replace(out)
        if input_meta.exists():
            shutil.copyfile(input_meta, output_meta)
        print(f"wrote {out}")
    finally:
        con.close()
