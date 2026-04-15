"""
RAGRetriever — lightweight Retrieval-Augmented Generation engine.

Indexes all CSV datasets as named passages (no external vector DB needed).
Retrieval uses domain-keyword scoring against the user query.
Each passage carries a [REF-N] citation that Qwen is instructed to use.

Author: Tuan Tran
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set

import pandas as pd


# ── Domain keyword vocabularies ───────────────────────────────────────────────
_DOMAIN_KW: Dict[str, Set[str]] = {
    "finance": {
        "finance", "financial", "revenue", "profit", "ebitda", "margin",
        "budget", "cash", "arr", "mrr", "churn", "cost", "expense", "opex",
        "tài chính", "doanh thu", "lợi nhuận", "chi phí", "ngân sách",
        "biên lợi nhuận", "dòng tiền",
    },
    "sales": {
        "sales", "product", "region", "channel", "units", "gross",
        "sold", "deal", "customer segment", "tier", "price",
        "bán hàng", "sản phẩm", "khu vực", "kênh bán", "đơn vị",
    },
    "hr": {
        "hr", "human", "headcount", "attrition", "employee", "hire",
        "enps", "engagement", "salary", "department", "training",
        "nhân sự", "nhân viên", "tuyển dụng", "nghỉ việc", "lương", "phòng ban",
    },
    "operations": {
        "ops", "operations", "efficiency", "sla", "defect", "deploy",
        "uptime", "incident", "ticket", "mttr", "dora",
        "vận hành", "hiệu suất", "sự cố", "triển khai",
    },
    "customers": {
        "customer", "nps", "csat", "retention", "ltv", "cac",
        "cohort", "churn rate", "arpu",
        "khách hàng", "giữ chân", "điểm nps",
    },
}


def _tokenize(text: str) -> Set[str]:
    """Word tokenizer: captures letters, digits, Vietnamese diacritics."""
    return set(re.findall(r'[\w\u00C0-\u1EF9]+', text.lower()))


def _domain_score(query: str, query_tokens: Set[str], domain: str) -> int:
    """
    Score how well the query matches a domain.
    - Single-word keywords: +1 per match in token set
    - Multi-word phrases:   +3 per match as substring in full query text
    """
    kw_set = _DOMAIN_KW.get(domain, set())
    ql = query.lower()
    total = 0
    for kw in kw_set:
        if ' ' in kw or '\u00C0' <= kw[0] <= '\u1EF9':  # phrase or Vietnamese
            if kw in ql:
                total += 3
        if kw in query_tokens:
            total += 1
    return total


def _compact(obj) -> str:
    """Compact JSON with no indentation."""
    return json.dumps(obj, separators=(",", ":"), default=str)


# ── Passage dataclass ──────────────────────────────────────────────────────────
@dataclass
class Passage:
    ref: str          # e.g. "[REF-3]"
    index: int        # 1-based
    domain: str       # "Finance" | "Sales" | "HR" | "Operations" | "Customers"
    source: str       # human-readable label, e.g. "financial_data.csv · Dec 2025"
    content: str      # compact data string for prompt injection
    keywords: Set[str] = field(default_factory=set)

    def prompt_block(self) -> str:
        return f"{self.ref} [{self.domain} · {self.source}]\n{self.content}"

    def citation_line(self) -> str:
        return f"{self.ref} {self.domain} — {self.source}"


# ── Main retriever ─────────────────────────────────────────────────────────────
class RAGRetriever:
    """
    Builds an in-memory passage index from the DataPipeline and retrieves
    the top-k most relevant passages for a given natural-language query.

    Usage
    -----
    retriever = RAGRetriever(pipeline)
    passages  = retriever.retrieve("doanh thu tháng 12 tăng như thế nào?", k=4)
    for p in passages:
        print(p.citation_line())
    """

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline
        self._passages: List[Passage] = []
        self._build_index()

    # ── Index construction ────────────────────────────────────
    def _build_index(self) -> None:
        idx = 0

        def add(domain: str, source: str, content: str,
                extra_kw: Set[str] | None = None) -> None:
            nonlocal idx
            idx += 1
            kw = _DOMAIN_KW.get(domain.lower(), set()).copy()
            if extra_kw:
                kw |= extra_kw
            self._passages.append(Passage(
                ref=f"[REF-{idx}]",
                index=idx,
                domain=domain,
                source=source,
                content=content,
                keywords=kw,
            ))

        p = self._pipeline

        # ── Finance ───────────────────────────────────────────
        try:
            fin = p.finance
            fin_cols = [c for c in
                        ["date", "revenue", "net_profit", "ebitda",
                         "gross_margin_pct", "ebitda_margin_pct",
                         "total_opex", "arr", "mrr", "churn_rate_pct"]
                        if c in fin.columns]
            # Recent 12 months, one passage each
            for _, row in fin.tail(12).iterrows():
                date_str = str(row["date"])[:7]
                data = {c: round(row[c], 2) if isinstance(row[c], float) else row[c]
                        for c in fin_cols}
                add("Finance", f"financial_data.csv · {date_str}", _compact(data),
                    extra_kw={date_str, str(row["date"].year)})
            # Overall finance summary
            summary = {
                "total_revenue": round(fin["revenue"].sum(), 0),
                "avg_monthly_revenue": round(fin["revenue"].mean(), 0),
                "total_net_profit": round(fin["net_profit"].sum(), 0),
                "avg_gross_margin_pct": round(fin["gross_margin_pct"].mean(), 1)
                    if "gross_margin_pct" in fin.columns else None,
                "period": f"{str(fin['date'].min())[:7]} to {str(fin['date'].max())[:7]}",
            }
            add("Finance", "financial_data.csv · Full Period Summary", _compact(summary))
        except Exception:
            pass

        # ── Sales ─────────────────────────────────────────────
        try:
            sal = p.sales_monthly
            sal_cols = [c for c in
                        ["date", "revenue", "units_sold", "gross_margin_pct"]
                        if c in sal.columns]
            for _, row in sal.tail(12).iterrows():
                date_str = str(row["date"])[:7]
                data = {c: round(row[c], 2) if isinstance(row[c], float) else row[c]
                        for c in sal_cols}
                add("Sales", f"sales_data.csv · Monthly {date_str}", _compact(data),
                    extra_kw={date_str})

            # By product (top 10)
            by_prod = p.sales_by_product.head(10)
            for _, row in by_prod.iterrows():
                add("Sales", f"sales_data.csv · Product: {row['product']}",
                    _compact(row.to_dict()),
                    extra_kw={str(row["product"]).lower()})

            # By region (all)
            by_region = p.sales_by_region
            for _, row in by_region.iterrows():
                add("Sales", f"sales_data.csv · Region: {row['region']}",
                    _compact(row.to_dict()),
                    extra_kw={str(row["region"]).lower()})
        except Exception:
            pass

        # ── HR ────────────────────────────────────────────────
        try:
            hr_m = p.hr_monthly
            hr_cols = [c for c in
                       ["date", "headcount", "new_hires", "attrition_rate",
                        "eNPS", "engagement_score"]
                       if c in hr_m.columns]
            for _, row in hr_m.tail(12).iterrows():
                date_str = str(row["date"])[:7]
                data = {c: round(row[c], 2) if isinstance(row[c], float) else row[c]
                        for c in hr_cols}
                add("HR", f"hr_data.csv · {date_str}", _compact(data),
                    extra_kw={date_str})

            # By department
            hc_col = "headcount_eop" if "headcount_eop" in p.hr.columns else "headcount"
            if "department" in p.hr.columns:
                dept_agg = p.hr.groupby("department").agg(
                    avg_headcount=(hc_col, "mean"),
                    avg_attrition=("attrition_rate_pct", "mean")
                    if "attrition_rate_pct" in p.hr.columns else (hc_col, "count"),
                ).reset_index()
                for _, row in dept_agg.iterrows():
                    add("HR", f"hr_data.csv · Dept: {row['department']}",
                        _compact(row.to_dict()),
                        extra_kw={str(row["department"]).lower()})
        except Exception:
            pass

        # ── Operations ────────────────────────────────────────
        try:
            ops = p.operations
            ops_cols = [c for c in
                        ["date", "process_efficiency_pct", "sla_compliance_pct",
                         "defect_rate_pct", "server_uptime_pct",
                         "deploy_frequency_month", "mean_time_to_restore_hrs",
                         "change_failure_rate_pct", "customer_satisfaction"]
                        if c in ops.columns]
            for _, row in ops.tail(12).iterrows():
                date_str = str(row["date"])[:7]
                data = {c: round(row[c], 2) if isinstance(row[c], float) else row[c]
                        for c in ops_cols}
                add("Operations", f"operations_data.csv · {date_str}", _compact(data),
                    extra_kw={date_str})
        except Exception:
            pass

        # ── Customers ─────────────────────────────────────────
        try:
            cust = p.customers
            if not cust.empty:
                cust_cols = [c for c in
                             ["date", "total_customers", "new_customers",
                              "churned_customers", "churn_rate_pct",
                              "retention_rate_pct", "nps_score", "csat_score",
                              "ltv_cac_ratio", "arpu"]
                             if c in cust.columns]
                for _, row in cust.tail(12).iterrows():
                    date_str = str(row["date"])[:7]
                    data = {c: round(row[c], 2) if isinstance(row[c], float) else row[c]
                            for c in cust_cols}
                    add("Customers", f"customer_data.csv · {date_str}", _compact(data),
                        extra_kw={date_str})
        except Exception:
            pass

    # ── Retrieval ─────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 5) -> List[Passage]:
        """
        Return up to k passages most relevant to *query*.

        Scoring:
        - +3 per domain keyword matched in query
        - +1 per passage-specific keyword matched
        - Recency boost for passages with date tokens that appear in query
        """
        if not query or not self._passages:
            return []

        query_tokens = _tokenize(query)

        scored: List[tuple[float, Passage]] = []
        n = len(self._passages)
        for p in self._passages:
            # Domain alignment (phrase-aware)
            domain_score = _domain_score(query, query_tokens, p.domain.lower()) * 3
            # Passage-specific keyword overlap
            kw_score = len(query_tokens & p.keywords)
            # Recency boost: passages indexed later are more recent (0..1 scale)
            recency = p.index / n
            score = domain_score + kw_score + recency
            if domain_score + kw_score > 0:
                scored.append((score, p))

        # Sort descending, take top k, re-number refs sequentially for the response
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [p for _, p in scored[:k]]

        # Re-assign sequential refs for this retrieval batch
        for i, passage in enumerate(top, start=1):
            passage.ref = f"[REF-{i}]"

        return top

    def __len__(self) -> int:
        return len(self._passages)
