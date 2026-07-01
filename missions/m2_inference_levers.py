"""M2 - Inference Cost Levers: $/1M-token, batch x cache x cascade.

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations

import os as _os
import sys as _sys
from collections import defaultdict

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from finops import pricing, sustainability
from missions._common import load_csv, num

# $/1M tokens (input, output) - illustrative June-2026 prices.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
CACHE_WRITE_COST_PER_M = {"small": 0.05, "large": 0.75}
REASONING_TRAFFIC_CAP = 0.05


def _cache_economics(rows: list[dict]) -> dict:
    """Estimate cached-prefix reuse by tier from cache-bearing project traffic."""
    groups = defaultdict(set)
    reads = defaultdict(int)
    for r in rows:
        cached = int(num(r["cached_input_tokens"]))
        if cached <= 0:
            continue
        tier = r["route_tier"]
        groups[tier].add((r["team"], r["project"]))
        reads[tier] += 1

    out = {}
    for tier, read_count in reads.items():
        unique_prefix_groups = max(1, len(groups[tier]))
        price_in, _ = MODEL_PRICES[tier]
        write_cost = CACHE_WRITE_COST_PER_M[tier]
        avg_reads = read_count / unique_prefix_groups
        break_even = pricing.cache_break_even_reads(write_cost, input_price_per_m=price_in)
        out[tier] = {
            "avg_cache_reads": round(avg_reads, 1),
            "break_even_reads": round(break_even, 2),
            "worth_it": pricing.cache_is_worth_it(avg_reads, write_cost, input_price_per_m=price_in),
        }
    return out


def _optimized_request_cost(r: dict, cache_policy: dict) -> float:
    inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
    cached = int(num(r["cached_input_tokens"]))
    is_batch = bool(int(num(r["is_batch"])))
    tier = r["route_tier"]
    pin, pout = MODEL_PRICES[tier]
    if not cache_policy.get(tier, {}).get("worth_it", False):
        cached = 0
    return pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)


def _reasoning_budget(rows: list[dict], cache_policy: dict, cap_frac: float = REASONING_TRAFFIC_CAP) -> dict:
    reasoning_cost = non_reasoning_cost = 0.0
    reasoning_wh = non_reasoning_wh = 0.0
    reasoning_count = non_reasoning_count = 0
    reasoning_tokens = non_reasoning_tokens = 0

    for r in rows:
        total = int(num(r["input_tokens"])) + int(num(r["output_tokens"]))
        cost = _optimized_request_cost(r, cache_policy)
        is_reasoning = bool(int(num(r["is_reasoning"])))
        wh = sustainability.wh_per_query(total, is_reasoning=is_reasoning)
        if is_reasoning:
            reasoning_count += 1
            reasoning_tokens += total
            reasoning_cost += cost
            reasoning_wh += wh
        else:
            non_reasoning_count += 1
            non_reasoning_tokens += total
            non_reasoning_cost += cost
            non_reasoning_wh += wh

    total_count = reasoning_count + non_reasoning_count
    total_tokens = reasoning_tokens + non_reasoning_tokens
    total_cost = reasoning_cost + non_reasoning_cost
    total_wh = reasoning_wh + non_reasoning_wh
    allowed_reasoning = int(total_count * cap_frac)
    excess = max(0, reasoning_count - allowed_reasoning)
    avg_reasoning_cost = reasoning_cost / reasoning_count if reasoning_count else 0.0
    avg_non_reasoning_cost = non_reasoning_cost / non_reasoning_count if non_reasoning_count else 0.0
    avg_reasoning_wh = reasoning_wh / reasoning_count if reasoning_count else 0.0
    cap_savings_cost = excess * max(0.0, avg_reasoning_cost - avg_non_reasoning_cost)
    cap_savings_wh = excess * avg_reasoning_wh * (1.0 - 1.0 / sustainability.REASONING_ENERGY_MULTIPLIER)

    return {
        "reasoning_requests": reasoning_count,
        "total_requests": total_count,
        "reasoning_traffic_pct": round(reasoning_count / total_count * 100, 1) if total_count else 0.0,
        "reasoning_token_pct": round(reasoning_tokens / total_tokens * 100, 1) if total_tokens else 0.0,
        "reasoning_cost_daily": round(reasoning_cost, 2),
        "reasoning_cost_pct": round(reasoning_cost / total_cost * 100, 1) if total_cost else 0.0,
        "reasoning_wh_daily": round(reasoning_wh, 1),
        "reasoning_wh_pct": round(reasoning_wh / total_wh * 100, 1) if total_wh else 0.0,
        "cap_target_pct": round(cap_frac * 100, 1),
        "cap_excess_requests": excess,
        "cap_savings_daily": round(cap_savings_cost, 2),
        "cap_savings_wh_daily": round(cap_savings_wh, 1),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    cache_policy = _cache_economics(rows)
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        total_tokens += inp + out

        # Baseline: naive deployment - everything on the large model, no cache, no batch.
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)

        # Optimized: cascade (route_tier), prompt caching when economical, batch API.
        opt_cost += _optimized_request_cost(r, cache_policy)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    reasoning = _reasoning_budget(rows, cache_policy)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("cache economics:")
        for tier, c in sorted(cache_policy.items()):
            print(
                f"  {tier:5} avg_reads={c['avg_cache_reads']:>5}  "
                f"break-even={c['break_even_reads']:>4}  worth_it={c['worth_it']}"
            )
        print(
            "reasoning budget: "
            f"{reasoning['reasoning_traffic_pct']:.1f}% traffic -> "
            f"{reasoning['reasoning_cost_pct']:.1f}% cost, {reasoning['reasoning_wh_pct']:.1f}% Wh; "
            f"cap at {reasoning['cap_target_pct']:.0f}% saves "
            f"${reasoning['cap_savings_daily']:.2f}/day and {reasoning['cap_savings_wh_daily']:.1f} Wh/day"
        )

    return {
        "baseline_daily": round(base_cost, 2),
        "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3),
        "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1),
        "total_tokens": total_tokens,
        "cache_policy": cache_policy,
        "reasoning_budget": reasoning,
    }


if __name__ == "__main__":
    run()
