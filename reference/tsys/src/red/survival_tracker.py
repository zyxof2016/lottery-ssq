from __future__ import annotations

from collections import Counter
from typing import Any


def red_survival_report(period_records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = Counter()
    candidate = Counter()
    method = Counter()
    final = Counter()
    missed_final = Counter()
    for record in period_records:
        red = ((record.get("candidate_recall_funnel") or {}).get("red") or {})
        actual = [int(value) for value in red.get("actual", []) or (record.get("actual") or {}).get("red", []) or []]
        for number in actual:
            eligible[number] += 1
        for number in actual:
            if number not in set(int(v) for v in red.get("missed_at_candidate_pool", []) or []):
                candidate[number] += 1
            if number not in set(int(v) for v in red.get("missed_at_method_layer", []) or []):
                method[number] += 1
            if number not in set(int(v) for v in red.get("missed_at_final_layer", []) or []):
                final[number] += 1
            else:
                missed_final[number] += 1
    rows = []
    for number in range(1, 34):
        total = eligible[number]
        if total <= 0:
            continue
        rows.append({
            "number": number,
            "eligible_periods": total,
            "candidate_survival_rate": candidate[number] / total,
            "method_survival_rate": method[number] / total,
            "final_survival_rate": final[number] / total,
            "final_miss_count": missed_final[number],
        })
    rows.sort(key=lambda item: (item["final_survival_rate"], -item["final_miss_count"], item["number"]))
    return {
        "schema": "red_survival_report_v7",
        "periods": len(period_records),
        "top_lost_numbers": rows[:10],
        "top_protected_numbers": sorted(rows, key=lambda item: item["final_survival_rate"], reverse=True)[:10],
        "explanation": "survival_rate(number, stage)=该真实号码通过该阶段次数/该号码实际出现次数",
    }
