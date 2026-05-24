import json

RESULTS_FILE = "results/benchmark_single_fold_results.json"

def print_markdown_table(headers, rows):
    # Determine column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, item in enumerate(row):
            widths[idx] = max(widths[idx], len(str(item)))
            
    # Print headers
    header_str = "| " + " | ".join(f"{str(h):<{widths[idx]}}" for idx, h in enumerate(headers)) + " |"
    print(header_str)
    
    # Print separator
    sep_str = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    print(sep_str)
    
    # Print rows
    for row in rows:
        row_str = "| " + " | ".join(f"{str(item):<{widths[idx]}}" for idx, item in enumerate(row)) + " |"
        print(row_str)

def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = data.get("results", {})
    
    datasets = list(results.keys())
    strategies = None
    for ds in datasets:
        if results[ds]:
            strategies = list(results[ds].keys())
            break
            
    if not strategies:
        print("No strategies found in results.")
        return
        
    print(f"Datasets: {datasets}")
    print(f"Strategies: {strategies}")
    
    # We will build a summary table for F1-score, Accuracy, Forest Size, and Convergence Round
    headers = ["Strategy"] + datasets
    
    for metric_name in ["f1", "accuracy", "forest_size", "convergence_round"]:
        print(f"\n### Metric: {metric_name.upper()}")
        rows = []
        for strat in strategies:
            row = [strat]
            for ds in datasets:
                strat_res = results[ds].get(strat, {})
                if strat_res.get("success"):
                    metrics = strat_res.get("metrics")
                    conv_rnd = strat_res.get("convergence_round")
                    if metric_name == "convergence_round":
                        val = conv_rnd if conv_rnd is not None else "N/A"
                    else:
                        val = metrics.get(metric_name) if metrics else None
                    
                    if isinstance(val, float):
                        row.append(f"{val:.4f}")
                    else:
                        row.append(str(val))
                else:
                    row.append("FAIL")
            rows.append(row)
            
        print_markdown_table(headers, rows)

if __name__ == "__main__":
    main()
