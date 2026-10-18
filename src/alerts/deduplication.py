"""Deduplication and State Management for Alerts."""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path

def apply_deduplication(candidates: List[Dict[str, Any]], state_file: Path, rules_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filters duplicate alerts and handles cooldowns."""
    now = datetime.utcnow()
    
    if state_file.exists():
        state_df = pd.read_csv(state_file)
        if 'timestamp' in state_df.columns:
            state_df['timestamp'] = pd.to_datetime(state_df['timestamp'])
    else:
        state_df = pd.DataFrame(columns=['alert_id', 'rule_id', 'region', 'category', 'severity', 'timestamp', 'status'])
        
    final_alerts = []
    
    for cand in candidates:
        rule_id = cand['rule_id']
        region = cand['region']
        
        # Deduplication key is rule_id + region
        
        cooldown_hours = rules_config.get(rule_id, {}).get('cooldown_hours', 24)
        
        # Check existing alerts for this key
        match = state_df[(state_df['rule_id'] == rule_id) & (state_df['region'] == region)]
        
        should_emit = True
        
        if not match.empty:
            last_alert = match.sort_values('timestamp').iloc[-1]
            last_time = pd.to_datetime(last_alert['timestamp'])
            
            # Check cooldown
            if now - last_time < timedelta(hours=cooldown_hours):
                # Inside cooldown period, don't emit unless severity escalated
                # Simplified: we suppress if inside cooldown
                should_emit = False
                
        if should_emit:
            # Generate a stable-ish alert ID
            cand['alert_id'] = f"ALT_{rule_id}_{region}_{now.strftime('%Y%md%H%M%S')}"
            cand['status'] = 'new'
            final_alerts.append(cand)
            
    return final_alerts
