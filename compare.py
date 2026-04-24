#!/usr/bin/env python3  
"""  
Compare latencies between two benchmark runs.  
  
Usage:  
    python compare_benchmark_latency.py run1.log run2.log  
"""  
  
import json  
import sys  
from typing import Dict, Tuple, List, Optional  
  
  
class BenchmarkMetrics:  
    """Simplified version of BenchmarkMetrics for comparison."""  
    def __init__(self, data: dict):  
        self.shape_detail = data.get("shape_detail", [])  
        self.latency = data.get("latency")  
        self.error_msg = data.get("error_msg")  
  
  
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
    shape_str = str(metric.shape_detail) if metric.shape_detail else "None"  
    return f"{result.op_name}_{result.dtype}_{result.mode}_{result.level}_{shape_str}"  
  
  
def compare_latencies(run1_results: List[BenchmarkResult],   
                     run2_results: List[BenchmarkResult]) -> Dict[str, Tuple[float, float, float]]:  
    """  
    Compare latencies between two runs.  
      
    Returns:  
        Dict mapping key to (latency1, latency2, percent_change)  
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
        lat1 = run1_lookup.get(key, None)  
        lat2 = run2_lookup.get(key, None)  
          
        if lat1 is not None and lat2 is not None:  
            percent_change = ((lat2 - lat1) / lat1) * 100  
            comparisons[key] = (lat1, lat2, percent_change)  
        elif lat1 is not None:  
            comparisons[key] = (lat1, None, -100.0)  # Missing in run2  
        elif lat2 is not None:  
            comparisons[key] = (None, lat2, 100.0)   # Missing in run1  
      
    return comparisons  
  
  
def print_comparison(comparisons: Dict[str, Tuple[float, float, float]]):  
    """Print comparison results in a formatted table."""  
    print("\nLatency Comparison Results:")  
    print("-" * 120)  
    print(f"{'Operation':<30} {'Shape':<25} {'Run1 (ms)':<12} {'Run2 (ms)':<12} {'Change (%)':<12}")  
    print("-" * 120)  
      
    # Group by operation using regular dict  
    by_op = {}  
    for key, (lat1, lat2, change) in comparisons.items():  
        parts = key.split('_')  
        op_name = '_'.join(parts[:-4])  # Reconstruct op name  
        shape = parts[-1]  
        if op_name not in by_op:  
            by_op[op_name] = []  
        by_op[op_name].append((shape, lat1, lat2, change))  
      
    for op_name in sorted(by_op.keys()):  
        for shape, lat1, lat2, change in sorted(by_op[op_name]):  
            lat1_str = f"{lat1:.6f}" if lat1 is not None else "N/A"  
            lat2_str = f"{lat2:.6f}" if lat2 is not None else "N/A"  
            change_str = f"{change:+.2f}" if lat1 is not None and lat2 is not None else "N/A"  
              
            print(f"{op_name:<30} {shape:<25} {lat1_str:<12} {lat2_str:<12} {change_str:<12}")  
      
    # Summary statistics  
    valid_changes = [change for _, (_, _, change) in comparisons.items()   
                    if change != -100.0 and change != 100.0]  
    if valid_changes:  
        avg_change = sum(valid_changes) / len(valid_changes)  
        max_increase = max(valid_changes)  
        max_decrease = min(valid_changes)  
          
        print("-" * 120)  
        print(f"Summary: {len(valid_changes)} comparisons")  
        print(f"Average change: {avg_change:+.2f}%")  
        print(f"Max increase: {max_increase:+.2f}%")  
        print(f"Max decrease: {max_decrease:+.2f}%")  
  
  
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
