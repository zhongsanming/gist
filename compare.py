#!/usr/bin/env python3
"""
Compare latencies between two benchmark runs.

Usage:
    python compare_benchmark_latency.py run1.log run2.log
"""

import json
import sys
from typing import Dict, Tuple, List, Optional

DISCARDED_METRICS  = ["latency_base", "gbps_base", "gbps", "speedup", "accuracy", "tflops", "utilization", "compared_speedup"]


class BenchmarkMetrics:
    """Simplified version of BenchmarkMetrics for comparison."""

    def __init__(self, data: dict):
        self.latency = data.get("latency")
        self.error_msg = data.get("error_msg")
        self.parameters = {k: v for k, v in data.items() if k not in DISCARDED_METRICS + ["error_msg", "latency"]}


class BenchmarkResult:
    """Simplified version of BenchmarkResult for comparison."""

    def __init__(self, data: dict):
        self.op_name = data["op_name"]
        self.dtype = data["dtype"]
        self.mode = data["mode"]
        self.level = data["level"]
        self.result = [BenchmarkMetrics(m) for m in data["result"]]


def parse_log(log_file_path: str) -> List[BenchmarkResult]:
    """Parse benchmark log file and return list of BenchmarkResult objects."""
    with open(log_file_path, "r") as file:
        log_lines = [
            line
            for line in file.read().strip().split("\n")
            if line.startswith("[INFO] {")
        ]

    benchmark_results = []
    for line in log_lines:
        if line.startswith("[INFO]"):
            json_str = line[len("[INFO] ") :]
            data = json.loads(json_str)
            benchmark_result = BenchmarkResult(data)
            benchmark_results.append(benchmark_result)

    return benchmark_results


def get_result_key(result: BenchmarkResult, metric: BenchmarkMetrics) -> str:
    """Generate unique key for matching results across runs."""
    parameter_str = "_".join(f"{k}_{v}" for k, v in metric.parameters.items())
    # Include dtype in the key for proper matching
    return result.op_name, result.dtype, result.mode, result.level, parameter_str
    # return f"{result.op_name}-{result.dtype}-{result.mode}-{result.level}-{parameter_str}"


def compare_latencies(
    run1_results: List[BenchmarkResult], run2_results: List[BenchmarkResult]
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compare latencies between two runs.

    Returns:
        Dict mapping key to (latency1, latency2, percent_change, shape_detail, dtype)
    """
    # Build lookup tables
    run1_lookup = {}
    for result in run1_results:
        for metric in result.result:
            if metric.latency is not None and metric.error_msg is None:
                key = get_result_key(result, metric)
                run1_lookup[key] = metric.latency

    run2_lookup = {}
    for result in run2_results:
        for metric in result.result:
            if metric.latency is not None and metric.error_msg is None:
                key = get_result_key(result, metric)
                run2_lookup[key] = metric.latency

    # Compare latencies
    comparisons = {}
    all_keys = set(run1_lookup.keys()) | set(run2_lookup.keys())

    for key in all_keys:
        run1_data = run1_lookup.get(key)
        run2_data = run2_lookup.get(key)

        if run1_data and run2_data:
            lat1 = run1_data
            lat2 = run2_data
            percent_change = ((lat2 - lat1) / lat1) * 100
            comparisons[key] = (lat1, lat2, percent_change)
        elif run1_data:
            lat1 = run1_data
            comparisons[key] = (lat1, None, -100.0)  # Missing in run2
        elif run2_data:
            lat2= run2_data
            comparisons[key] = (None, lat2, 100.0)  # Missing in run1

    return comparisons


def print_comparison(comparisons: Dict[str, Tuple[float, float, float]]):
    """Print comparison results in a formatted table."""
    print("\n" + "=" * 140)
    print(
        f"{'Operation':<25} {'Run1 (ms)':<12} {'Run2 (ms)':<12} {'Change (%)':<12}   {'Parameters':<70}"
    )
    print("=" * 140)

    # Group by operation using regular dict
    by_op = {}
    for (op_name, dtype, mode, level, parameters), (lat1, lat2, change) in comparisons.items():
        if op_name not in by_op:
            by_op[op_name] = []
        by_op[op_name].append((lat1, lat2, change, f"{dtype}_{parameters}"))

    for op_name in sorted(by_op.keys()):
        # Print operation header
        print(f"\n{op_name}:")
        print("-" * 140)

        # Sort by shape for consistent ordering
        for lat1, lat2, change, parameters in sorted(
            by_op[op_name], key=lambda x: str(x[3])
        ):
            lat1_str = f"{lat1:.6f}" if lat1 is not None else "N/A"
            lat2_str = f"{lat2:.6f}" if lat2 is not None else "N/A"

            if lat1 is not None and lat2 is not None:
                change_str = f"{change:+8.2f}"
                # Intuitive arrow indicators (equal width)
                if change > 0:
                    change_str = f"↓↓↓ {change_str}"
                elif change < 0:
                    change_str = f"↑↑↑ {change_str}"
                else:
                    change_str = f"─── {change_str}"
            else:
                change_str = "N/A      "

            print(
                f"{'':<25} {lat1_str:<12} {lat2_str:<12} {change_str:<12} {parameters:<70}"
            )

    # Summary statistics
    valid_changes = [
        change
        for _, (_, _, change) in comparisons.items()
        if change != -100.0 and change != 100.0
    ]

    # Top 10 performance decreases
    decreases = [
        (key, change)
        for key, (_, _, change) in comparisons.items()
        if change > 0 and change != 100.0
    ]
    decreases.sort(key=lambda x: x[1], reverse=True)  # Sort by largest decrease

    if valid_changes:
        avg_change = sum(valid_changes) / len(valid_changes)
        max_increase = max(valid_changes)
        max_decrease = min(valid_changes)

        print("\n" + "=" * 140)
        print(f"SUMMARY STATISTICS:")
        print(f"  Total comparisons: {len(valid_changes)}")
        print(f"  Average change: {avg_change:+8.2f}%")
        print(f"  Max increase:   {max_increase:+8.2f}% ↓↓↓")
        print(f"  Max decrease:   {max_decrease:+8.2f}% ↑↑↑")

        # Top 10 performance decreases
        if decreases:
            print(f"\nTOP 10 PERFORMANCE DECREASES:")
            print("-" * 140)
            print(
                f"{'Rank':<6} {'Operation':<25} {'Change (%)':<12} {'Shape & Dtype':<70}"
            )
            print("-" * 140)

            for rank, ((op_name, dtype, mode, level, parameter_str), change) in enumerate(decreases[:10], 1):
                parameters = f"{dtype}_{parameter_str}"
                print(
                    f"{rank:<6} {op_name:<25} {change:+8.2f}%{'':<4} {parameters:<70}"
                )

        print("=" * 140)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_benchmark_latency.py <run1.log> <run2.log>")
        sys.exit(1)

    run1_file = sys.argv[1]
    run2_file = sys.argv[2]

    print(f"Parsing {run1_file}...")
    run1_results = parse_log(run1_file)

    print(f"Parsing {run2_file}...")
    run2_results = parse_log(run2_file)

    print(f"Run 1: {len(run1_results)} operations")
    print(f"Run 2: {len(run2_results)} operations")

    print("\nComparing latencies...")
    comparisons = compare_latencies(run1_results, run2_results)

    print_comparison(comparisons)


if __name__ == "__main__":
    main()
