import pandas as pd

try:
    df = pd.read_csv('results/results_master_76_4_20.csv')
    # Group by strategy and calculate the mean of metrics
    summary = df.groupby('strategy')[['avg_f1', 'avg_accuracy', 'avg_pcd']].mean().reset_index()
    
    # Sort by avg_f1 descending to show best strategies first
    summary = summary.sort_values(by='avg_f1', ascending=False)
    
    print(summary.to_markdown(index=False))
except Exception as e:
    print(f"Error: {e}")
