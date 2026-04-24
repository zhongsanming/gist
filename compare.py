
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
    # Include dtype in the key for proper matching  
    return f"{result.op_name}_{result.dtype}_{result.mode}_{result.level}_{shape_str}"  
  
  
def parse_key(key: str) -> Tuple[str, str, str, str, str]:  
    """Parse the key back into its components."""  
    # Known values for mode and level  
    known_modes = ["kernel", "operator", "wrapper"]  
    known_levels = ["core", "comprehensive"]  
    known_dtypes = ["float16", "float32", "bfloat16", "int16", "int32", "bool", "cfloat"]  
      
    # Split by underscore  
    parts = key.split('_')  
      
    # Find dtype (first known dtype in the list)  
    dtype_idx = None  
    for i, part in enumerate(parts):  
        if part in known_dtypes:  
            dtype_idx = i  
            break  
      
    if dtype_idx is None:  
        # Fallback: assume format op_dtype_mode_level_shape  
        return parts[0], parts[1], parts[2], parts[3], '_'.join(parts[4:])  
      
    # Extract components  
    op_name = '_'.join(parts[:dtype_idx])  
    dtype = parts[dtype_idx]  
      
    # Mode should be right after dtype  
    mode_idx = dtype_idx + 1  
    mode = parts[mode_idx] if mode_idx < len(parts) else "kernel"  
      
    # Level should be right after mode  
    level_idx = mode_idx + 1  
    level = parts[level_idx] if level_idx < len(parts) else "core"  
      
    # Shape is everything after level  
    shape = '_'.join(parts[level_idx + 1:]) if level_idx + 1 < len(parts) else "None"  
      
    return op_name, dtype, mode, level, shape  
  
  
def format_shape_with_dtype(shape_detail: List, dtype: str) -> str:  
    """Format shape detail with dtype for display."""  
    shape_str = str(shape_detail) if shape_detail else "N/A"  
    # Extract short dtype name (e.g., "float16" from "torch.float16")  
    dtype_short = dtype.split(".")[-1] if "." in dtype else dtype  
    return f"{shape_str} ({dtype_short})"  
  
  
def compare_latencies(run1_results: List[BenchmarkResult],   
                     run2_results: List[BenchmarkResult]) -> Dict[str, Tuple[float, float, float, str, str]]:  
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
                run1_lookup[key] = (metric.latency, metric.shape_detail, result.dtype)  
      
    run2_lookup = {}  
    for result in run2_results:  
        for metric in result.result:  
            if metric.latency is not None and metric.error_msg is None:  
                key = get_result_key(result, metric)  
                run2_lookup[key] = (metric.latency, metric.shape_detail, result.dtype)  
      
    # Compare latencies  
    comparisons = {}  
    all_keys = set(run1_lookup.keys()) | set(run2_lookup.keys())  
      
    for key in all_keys:  
        run1_data = run1_lookup.get(key)  
        run2_data = run2_lookup.get(key)  
          
        if run1_data and run2_data:  
            lat1, shape1, dtype1 = run1_data  
            lat2, shape2, dtype2 = run2_data  
            percent_change = ((lat2 - lat1) / lat1) * 100  
            comparisons[key] = (lat1, lat2, percent_change, shape1, dtype1)  
        elif run1_data:  
            lat1, shape1, dtype1 = run1_data  
            comparisons[key] = (lat1, None, -100.0, shape1, dtype1)  # Missing in run2  
        elif run2_data:  
            lat2, shape2, dtype2 = run2_data  
            comparisons[key] = (None, lat2, 100.0, shape2, dtype2)   # Missing in run1  
      
    return comparisons  
  
  
def print_comparison(comparisons: Dict[str, Tuple[float, float, float, List, str]]):  
    """Print comparison results in a formatted table."""  
    print("\n" + "=" * 140)  
    print(f"{'Operation':<25} {'Run1 (ms)':<12} {'Run2 (ms)':<12} {'Change (%)':<12} {'Shape & Dtype':<70}")  
    print("=" * 140)  
      
    # Group by operation using regular dict  
    by_op = {}  
    for key, (lat1, lat2, change, shape, dtype) in comparisons.items():  
        # Parse the key to extract op_name correctly  
        op_name, _, _, _, _ = parse_key(key)  
        if op_name not in by_op:  
            by_op[op_name] = []  
        by_op[op_name].append((key, lat1, lat2, change, shape, dtype))  
      
    for op_name in sorted(by_op.keys()):  
        # Print operation header  
        print(f"\n{op_name}:")  
        print("-" * 140)  
          
        # Sort by shape for consistent ordering  
        for key, lat1, lat2, change, shape, dtype in sorted(by_op[op_name], key=lambda x: str(x[4])):  
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
              
            shape_dtype_str = format_shape_with_dtype(shape, dtype)  
              
            print(f"{'':<25} {lat1_str:<12} {lat2_str:<12} {change_str:<12} {shape_dtype_str:<70}")  
      
    # Summary statistics  
    valid_changes = [change for _, (_, _, change, _, _) in comparisons.items()   
                    if change != -100.0 and change != 100.0]  
      
    # Top 10 performance decreases  
    decreases = [(key, change) for key, (_, _, change, _, _) in comparisons.items()   
                if change > 0 and change != 100.0]  
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
            print(f"{'Rank':<6} {'Operation':<25} {'Change (%)':<12} {'Shape & Dtype':<70}")  
            print("-" * 140)  
              
            for rank, (key, change) in enumerate(decreases[:10], 1):  
                # Parse the key to extract op_name correctly  
                op_name, _, _, _, shape_str = parse_key(key)  
                # Extract dtype from the parsed key  
                _, dtype, _, _, _ = parse_key(key)  
                shape_dtype_str = format_shape_with_dtype(eval(shape_str), dtype)  
                  
                print(f"{rank:<6} {op_name:<25} {change:+8.2f}%{'':<4} {shape_dtype_str:<70}")  
          
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
