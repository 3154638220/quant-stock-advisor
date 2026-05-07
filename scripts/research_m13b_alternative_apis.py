"""M13-B 替代 API 调研：个股-概念绑定数据源可行性评估。

执行日期: 2026-05-07（更新: 同日完成数据回填）
最终结论: EM akshare stock_board_concept_cons_em() API 可用！
          486 概念 / 67,487 行 / 5,593 独特股票，100% 覆盖率。
          Tushare Pro 免费 token 权限不足，不再需要。

用法:
  python scripts/research_m13b_alternative_apis.py          # 完整调研报告
  python scripts/research_m13b_alternative_apis.py --fetch  # 拉取概念列表并写入 DuckDB
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date

import requests

_LOG = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1. 调研结果速查
# ═══════════════════════════════════════════════════════════════════

RESEARCH_SUMMARY = """
================================================================================
M13-B 替代 API 调研 — 结论速查（2026-05-07 最终更新）
================================================================================

✅ 问题已解决！EM akshare stock_board_concept_cons_em() API 可用。
   486 概念全部回填成功，67,487 行，5,593 只独特股票，0 失败。

根本原因: 之前失败是因为使用了 THS 概念名（如 "AI PC"），
          而 EM API 需要 EM 概念名（如 "C2M概念"）。
          EM sidemenu API 提供的概念名与 stock_board_concept_cons_em() 一致。

┌─────────────────────┬──────────┬────────────────────────────────────────────┐
│ 数据源              │ 状态     │ 详情                                       │
├─────────────────────┼──────────┼────────────────────────────────────────────┤
│ EM akshare          │ 🟢 可用  │ 486 概念/67K 行，100% 覆盖，0 失败         │
│ stock_board_concept │          │ 关键：必须使用 EM 概念名（sidemenu API），  │
│ _cons_em()          │          │ 不能使用 THS 概念名                         │
├─────────────────────┼──────────┼────────────────────────────────────────────┤
│ EM sidemenu API     │ ✅ 可用  │ 获取 EM 概念名称+代码（1,014 板块）         │
├─────────────────────┼──────────┼────────────────────────────────────────────┤
│ EM push2 clist/get  │ ❌ 阻断  │ TCP 层 RST（GFW DPI），不可修复             │
├─────────────────────┼──────────┼────────────────────────────────────────────┤
│ Tushare Pro         │ ⚠️ 不足  │ 免费 token 无概念 API 权限                  │
├─────────────────────┼──────────┼────────────────────────────────────────────┤
│ 雪球 Xueqiu         │ ❌ 需认证│ 需 xq_a_token                              │
├─────────────────────┼──────────┼────────────────────────────────────────────┤
│ Wind API            │ ⚠️ 商业  │ 需付费订阅，数据最全（含历史快照）          │
└─────────────────────┴──────────┴────────────────────────────────────────────┘

下一步:
  1. ✅ 数据管道已通 → 每月末随 SOP 做概念快照累积
  2. 🔄 概念因子开发 → 接入 M5 gate，验证 delta vs hard-cap baseline
  3. 🔄 历史 PIT 快照 → 需 ≥ 6 月累积后用于完整 walk-forward 回测

命名兼容性:
  EM 名 vs THS 名不同！backfill 必须用 EM 名。concept_client.py 已修复，
  优先从 a_share_concept_meta（EM sidemenu 数据）读取概念名。
================================================================================
"""

# ═══════════════════════════════════════════════════════════════════
# 2. 可用功能：EM sidemenu 概念列表拉取
# ═══════════════════════════════════════════════════════════════════

SIDEMENU_URL = "https://quote.eastmoney.com/center/api/sidemenu_new.json"


def fetch_concept_list_from_sidemenu() -> list[dict]:
    """从 EM sidemenu API 获取全量概念板块名称与代码。

    返回 list[dict]，每个 dict 包含:
      - market: int      市场代码（90 = 概念板块）
      - code: str        BK 代码，如 "BK06551"
      - name: str        板块名称，如 "融资融券"
      - pinyin: str      拼音缩写
      - type: int        板块类型（1=地域, 2=行业, 3=概念）

    可用性: ✅ 已验证（2026-05-07），HTTP 200，~99KB JSON，486 个概念。
    覆盖率: 100%（同一数据源与已封禁的 push2 API 共享板块代码体系）。
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://quote.eastmoney.com/center/boardlist.html",
    })

    resp = session.get(SIDEMENU_URL, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    bklist = data.get("bklist", [])
    if not bklist:
        raise RuntimeError("sidemenu_new.json 返回空 bklist")

    return [
        {
            "market": int(item.get("market", 0)),
            "code": str(item.get("code", "")),
            "name": str(item.get("name", "")),
            "pinyin": str(item.get("pinyin", "")),
            "type": int(item.get("type", 0)),
        }
        for item in bklist
        if item.get("code", "").startswith("BK")
    ]


def write_concept_list_to_db(conn, concepts: list[dict]) -> dict:
    """将概念列表写入 DuckDB a_share_concept_meta 表。"""
    import pandas as pd

    today = date.today()
    df = pd.DataFrame(concepts)
    df["concept_code"] = df["code"]
    df["concept_name"] = df["name"]
    df["first_seen_date"] = today
    df["stock_count"] = None
    df["total_mv"] = None

    conn.execute("DELETE FROM a_share_concept_meta")
    conn.execute(
        """
        INSERT INTO a_share_concept_meta
        (concept_code, concept_name, stock_count, total_mv, first_seen_date)
        SELECT concept_code, concept_name, stock_count, total_mv, first_seen_date
        FROM df
        """,
    )

    by_type = df["type"].value_counts().to_dict() if "type" in df.columns else {}
    stats = {
        "total_concepts": len(concepts),
        "type_1_region": int(by_type.get(1, 0)),
        "type_2_industry": int(by_type.get(2, 0)),
        "type_3_concept": int(by_type.get(3, 0)),
    }
    _LOG.info("concept meta written: %s", stats)
    return stats


# ═══════════════════════════════════════════════════════════════════
# 3. Tushare 接入模板（待 token 后启用）
# ═══════════════════════════════════════════════════════════════════

TUSHARE_CONCEPT_DETAIL_INSTRUCTIONS = """
# Tushare Pro 概念成分接入（已弃用，仅保留参考）

## 结论（2026-05-07）
  免费 token 无法调用 concept_detail / ths_concept / ths_member 等概念 API。
  所有概念成分 API 返回: "抱歉，您没有接口访问权限"。
  不再依赖 Tushare — EM akshare stock_board_concept_cons_em() 已满足需求。

## 权限状态（2026-05-07 实测）
  - concept_detail:  无权限
  - ths_concept:      API 不存在
  - ths_member:      无权限
  - concept:         无权限
  - ths_daily:       无权限

## 如需未来使用（如 Wind 不可用且需更高数据质量）
  1. 访问 https://tushare.pro 升级积分（需 ≥ 2000 分）
  2. 积分获取方式：付费或社区贡献
  3. 高积分后 concept_detail 可提供历史变更记录（in_date/out_date）
"""

# ═══════════════════════════════════════════════════════════════════
# 4. 手工维护兜底方案
# ═══════════════════════════════════════════════════════════════════

MANUAL_CONCEPT_PLAN = """
# 手工概念成分维护方案（兜底，如 Tushare 积分不足）

## 范围
  仅维护交易活跃度高、资金关注度高的 Top-50 概念板块。
  选择标准：近 1 月涨幅排名前 50 的概念板块。

## 操作流程
  1. 每月末从 EM data.eastmoney.com/bkzj/gn.html 获取热门概念排名
  2. 对每个概念，手工查看 EM 概念详情页获取前 20 大权重股
  3. 写入 a_share_concept_membership 表（snapshot_date = 月末日期）

## 维护成本
  每概念约 3 分钟 × 50 个 = 2.5 小时/月

## 覆盖度预期
  Top-50 概念覆盖 U1 池约 40-60%（热门概念包含大部分活跃标的）
"""

# ═══════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if "--fetch" in sys.argv:
        _LOG.info("fetching concept list from EM sidemenu API ...")
        concepts = fetch_concept_list_from_sidemenu()
        _LOG.info(
            "fetched %d concepts (type 1=%d, type 2=%d, type 3=%d)",
            len(concepts),
            sum(1 for c in concepts if c["type"] == 1),
            sum(1 for c in concepts if c["type"] == 2),
            sum(1 for c in concepts if c["type"] == 3),
        )

        import duckdb

        db_path = "data/market.duckdb"
        conn = duckdb.connect(db_path, read_only=False)
        try:
            stats = write_concept_list_to_db(conn, concepts)
            _LOG.info("written to %s: %s", db_path, stats)
        finally:
            conn.close()

        # Print sample concepts
        print("\nSample concepts (type=3):")
        type3 = [c for c in concepts if c["type"] == 3]
        for c in type3[:10]:
            print(f"  {c['code']}: {c['name']}")
        print(f"  ... and {len(type3) - 10} more")

    else:
        print(RESEARCH_SUMMARY)
        print()
        print("ℹ️  Tushare 路径已弃用（免费 token 权限不足），见下方说明。")
        print(TUSHARE_CONCEPT_DETAIL_INSTRUCTIONS)
        print()
        print(MANUAL_CONCEPT_PLAN)


if __name__ == "__main__":
    main()
