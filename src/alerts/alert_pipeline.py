"""Alert Execution Pipeline."""
import logging
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

from src.alerts.config import get_alert_config
from src.alerts.rule_evaluator import evaluate_rules
from src.alerts.deduplication import apply_deduplication

logger = logging.getLogger(__name__)

def load_alert_data(config_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    datasets = {}
    for key, path_str in config_paths.items():
        if not path_str:
            datasets[key] = None
            continue
        p = Path(path_str)
        if p.exists() and p.suffix == '.csv':
            try:
                datasets[key] = pd.read_csv(p)
            except Exception:
                datasets[key] = None
        else:
            datasets[key] = None
    return datasets

def run_alert_evaluation():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Intelligent Alerting Engine (Phase 13)")
    
    config = get_alert_config()
    region = config.get("alerts.region", "Bhopal")
    
    out_dir = Path(config.get("alerts.outputs.data_dir", "data/alerts"))
    reports_dir = Path(config.get("alerts.outputs.reports_dir", "reports/alerts"))
    for d in [out_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    state_file = out_dir / "alerts.csv"
    
    data_paths = config.get("alerts.data", {})
    datasets = load_alert_data(data_paths)
    
    rules = config.get("alerts.rules", {})
    
    # 1. Evaluate rules
    candidates = evaluate_rules(datasets, region, rules)
    
    # 2. Deduplicate
    new_alerts = apply_deduplication(candidates, state_file, rules)
    
    # 3. Save new alerts
    if new_alerts:
        new_df = pd.DataFrame(new_alerts)
        if state_file.exists():
            existing = pd.read_csv(state_file)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(state_file, index=False)
        
        # Save evidence snapshot
        evidence_df = new_df[['alert_id', 'rule_id', 'region', 'triggering_metric', 'observed_value', 'threshold', 'timestamp']]
        ev_file = out_dir / "alert_evidence.csv"
        if ev_file.exists():
            ev_existing = pd.read_csv(ev_file)
            pd.concat([ev_existing, evidence_df], ignore_index=True).to_csv(ev_file, index=False)
        else:
            evidence_df.to_csv(ev_file, index=False)
            
        logger.info(f"Generated {len(new_alerts)} new alerts.")
    else:
        logger.info("No new alerts generated.")
        
    # 4. Generate report
    if state_file.exists():
        state_df = pd.read_csv(state_file)
        with open(reports_dir / "alert_summary.md", "w") as f:
            f.write("# Alert Summary Report\n\n")
            f.write(f"**Region:** {region}\n\n")
            f.write("## Active Alerts\n")
            f.write(f"Total historical alerts: {len(state_df)}\n")
            f.write(f"New alerts in this run: {len(new_alerts)}\n\n")
            
            f.write("### New Alerts Details\n")
            for a in new_alerts:
                f.write(f"- **{a['alert_id']}** [{a['severity']}]: {a['explanation']}\n")
                
    logger.info("Alert Evaluation Pipeline completed.")

if __name__ == "__main__":
    from typing import Dict
    run_alert_evaluation()
