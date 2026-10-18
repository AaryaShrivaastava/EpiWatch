"""Visualizations for spatial data."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_incidence_comparison(df: pd.DataFrame, out_path: Path):
    """Plot bar chart of incidence."""
    if 'region' not in df.columns or 'incidence_per_100k' not in df.columns:
        return
        
    plt.figure(figsize=(10,6))
    valid = df.dropna(subset=['incidence_per_100k']).sort_values('incidence_per_100k', ascending=False)
    if valid.empty:
        return
        
    plt.bar(valid['region'], valid['incidence_per_100k'], color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Incidence per 100k')
    plt.title('Regional Incidence Comparison')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    
def plot_stress_vs_incidence(df: pd.DataFrame, out_path: Path):
    """Scatter plot."""
    if 'healthcare_stress_score' not in df.columns or 'incidence_per_100k' not in df.columns:
        return
        
    plt.figure(figsize=(8,6))
    valid = df.dropna(subset=['incidence_per_100k', 'healthcare_stress_score'])
    if valid.empty:
        return
        
    plt.scatter(valid['incidence_per_100k'], valid['healthcare_stress_score'], alpha=0.7)
    
    # Add labels
    for i, row in valid.iterrows():
        plt.annotate(row['region'], (row['incidence_per_100k'], row['healthcare_stress_score']))
        
    plt.xlabel('Incidence per 100k')
    plt.ylabel('Healthcare Stress Score')
    plt.title('Healthcare Stress vs Disease Incidence')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
