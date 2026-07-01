"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 inference: dict | None = None, efficiency: dict | None = None,
                 extensions: dict | None = None) -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if inference:
        lines += [
            "",
            "## Unit Economics",
            "",
            f"- Baseline inference: ${inference.get('baseline_per_m', 0):.3f}/1M-token",
            f"- Optimized inference: ${inference.get('optimized_per_m', 0):.3f}/1M-token",
            f"- Inference savings: {inference.get('savings_pct', 0):.1f}% from cascade, cache, and batch",
        ]
    if efficiency:
        lies = efficiency.get("lies", [])
        lie_ids = ", ".join(l.get("gpu_id", "unknown") for l in lies) or "none"
        lines += [
            "",
            "## Efficiency Audit",
            "",
            f"- GPU-Util lies flagged: {lie_ids}",
            f"- Idle waste identified: ${efficiency.get('idle_waste_daily', 0):,.2f}/day",
            "- GPU-Util only shows that clocks were active; MFU/MBU show whether rented FLOPs and HBM bandwidth created useful work.",
        ]
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
    if extensions:
        lines += ["", "## Your Turn Extensions", ""]
        cache = extensions.get("cache_policy", {})
        if cache:
            lines += ["### Cache Economics", ""]
            for tier, c in sorted(cache.items()):
                lines.append(
                    f"- {tier}: avg cached reads {c['avg_cache_reads']} vs. "
                    f"break-even {c['break_even_reads']} -> worth it: {c['worth_it']}"
                )
        reasoning = extensions.get("reasoning_budget", {})
        if reasoning:
            lines += [
                "",
                "### Reasoning Budget",
                "",
                f"- Reasoning is {reasoning.get('reasoning_traffic_pct', 0):.1f}% of requests and "
                f"{reasoning.get('reasoning_token_pct', 0):.1f}% of tokens, but "
                f"{reasoning.get('reasoning_cost_pct', 0):.1f}% of inference cost and "
                f"{reasoning.get('reasoning_wh_pct', 0):.1f}% of inference energy.",
                f"- Cap reasoning at {reasoning.get('cap_target_pct', 0):.0f}% of traffic; route only low-confidence or high-complexity tasks to reasoning.",
                f"- Estimated cap savings: ${reasoning.get('cap_savings_daily', 0):.2f}/day and "
                f"{reasoning.get('cap_savings_wh_daily', 0):.1f} Wh/day.",
            ]
    lines += [
        "",
        "## Recommended Actions",
        "",
        "1. Apply inference routing first: cascade simple traffic, keep prompt caching on for reused prefixes, and batch latency-tolerant eval traffic.",
        "2. Move interruptible training to spot with checkpointing, and reserve only steady workloads above the break-even duty cycle.",
        "3. Retire idle GPUs and right-size GPU-Util lies after confirming MFU/MBU bottlenecks.",
    ]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Savings (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
