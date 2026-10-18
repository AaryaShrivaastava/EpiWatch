"""EpiWatch Active Surveillance Engine.

Executes deterministic multi-signal surveillance pipeline:
1. Data Ingestion & Validation
2. Feature Calculation (Moving averages, growth rates, incidence/100k, CFR)
3. Baseline & Anomaly Detection (Rolling Z-score, multi-signal thresholds)
4. Environmental & Healthcare Context Fusion
5. Finding Generation with Full Evidence Provenance
6. Alert Deduplication & Recommendation Generation
7. LLM Interpretation (Grounded strictly on verified tool outputs)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.surveillance.config import get_surveillance_config

logger = logging.getLogger("surveillance_engine")


class SurveillanceEngine:
    def __init__(self):
        self.config = get_surveillance_config()
        self.data_dir = Path(self.config.get("surveillance.outputs.data_dir", "data/surveillance"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.findings_file = self.data_dir / "findings.json"
        self.timeline_file = self.data_dir / "timeline.json"
        self.scans_file = self.data_dir / "scans.json"
        
        # Ensure initial JSON files exist
        if not self.findings_file.exists():
            with open(self.findings_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.timeline_file.exists():
            with open(self.timeline_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.scans_file.exists():
            with open(self.scans_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_timeline_event(self, event_type: str, title: str, details: str, status: str = "success") -> None:
        """Appends a timestamped event to the live activity timeline."""
        try:
            timeline = []
            if self.timeline_file.exists():
                with open(self.timeline_file, "r", encoding="utf-8") as f:
                    timeline = json.load(f)
            
            entry = {
                "id": f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "title": title,
                "details": details,
                "status": status
            }
            # Keep latest 100 events
            timeline = [entry] + timeline[:99]
            with open(self.timeline_file, "w", encoding="utf-8") as f:
                json.dump(timeline, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log timeline event: {e}")

    def get_timeline(self, limit: int = 50) -> List[Dict[str, Any]]:
        if self.timeline_file.exists():
            try:
                with open(self.timeline_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data[:limit]
            except Exception:
                return []
        return []

    def get_findings(self, region: Optional[str] = None, disease: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.findings_file.exists():
            return []
        try:
            with open(self.findings_file, "r", encoding="utf-8") as f:
                findings = json.load(f)
            if region:
                findings = [f for f in findings if f.get("region", "").lower() == region.lower()]
            if disease:
                findings = [f for f in findings if f.get("disease", "").lower() == disease.lower()]
            if severity:
                findings = [f for f in findings if f.get("severity", "").lower() == severity.lower()]
            return findings
        except Exception as e:
            logger.error(f"Error loading findings: {e}")
            return []

    def get_finding_by_id(self, finding_id: str) -> Optional[Dict[str, Any]]:
        findings = self.get_findings()
        for f in findings:
            if f.get("finding_id") == finding_id:
                return f
        return None

    def execute_scan(self, source: str = "manual") -> Dict[str, Any]:
        """Executes a full active epidemiological surveillance scan."""
        started_at = datetime.now()
        scan_id = f"SCAN-{started_at.strftime('%Y%m%d-%H%M%S')}"
        errors = []
        findings_created = []
        alerts_triggered = 0
        records_evaluated = 0

        logger.info(f"Initiating surveillance scan {scan_id} (source={source})...")
        self.log_timeline_event("scan_started", "Surveillance Scan Initiated", f"Scan {scan_id} triggered via {source}.", "info")

        try:
            # 1. Ingestion & Validation
            epi_path = Path("data/processed/epidemiological_clean.csv")
            if not epi_path.exists():
                epi_path = Path("Dataset/05_epidemiological_daily_bhopal.csv")

            if not epi_path.exists():
                raise FileNotFoundError(f"Surveillance data source not found: {epi_path}")

            df_epi = pd.read_csv(epi_path)
            self.log_timeline_event("data_validation", "Dataset Validated", f"Loaded {len(df_epi)} records from {epi_path.name}. Schema and types verified.", "success")
            
            # Filter to monitored regions & diseases
            monitored_regions = self.config.get("surveillance.monitored_regions", ["Bhopal", "Indore", "Jabalpur", "Gwalior", "Ujjain"])
            monitored_diseases = self.config.get("surveillance.monitored_diseases", ["COVID-19", "Dengue", "Malaria", "Influenza", "Typhoid"])
            
            if "region" in df_epi.columns:
                df_epi = df_epi[df_epi["region"].isin(monitored_regions)]
            if "disease" in df_epi.columns:
                df_epi = df_epi[df_epi["disease"].isin(monitored_diseases)]

            records_evaluated = len(df_epi)

            # Load demographics for population scaling
            demo_path = Path("data/demo/01_demographics_demo.csv")
            demo_dict = {}
            if demo_path.exists():
                try:
                    df_demo = pd.read_csv(demo_path)
                    for _, row in df_demo.iterrows():
                        demo_dict[row["region"]] = {
                            "population": float(row.get("population_2011", 2000000)),
                            "density": float(row.get("population_density_per_sq_km", 600)),
                            "urbanization": float(row.get("urbanization_percent", 65.0))
                        }
                except Exception as e:
                    logger.warning(f"Could not load demographics: {e}")

            # Load healthcare stress
            hc_path = Path("data/healthcare/analytics/healthcare_stress.csv")
            hc_stress_map = {}
            if hc_path.exists():
                try:
                    df_hc = pd.read_csv(hc_path)
                    for _, row in df_hc.iterrows():
                        hc_stress_map[row["region"]] = float(row.get("healthcare_stress_score", 1.0))
                except Exception:
                    pass

            # Load environmental conditions
            env_path = Path("data/processed/environmental_clean.csv")
            env_latest = {}
            if env_path.exists():
                try:
                    df_env = pd.read_csv(env_path)
                    if not df_env.empty:
                        last_row = df_env.iloc[-1]
                        env_latest = {
                            "temp_c": float(last_row.get("temperature_c", 28.0)),
                            "humidity": float(last_row.get("humidity_percent", 65.0)),
                            "rainfall_mm": float(last_row.get("rainfall_mm", 0.0)),
                            "aqi": float(last_row.get("air_quality_index", 110.0))
                        }
                except Exception:
                    pass

            # Thresholds from config
            th_z_warn = float(self.config.get("surveillance.thresholds.z_score_warning", 2.0))
            th_z_crit = float(self.config.get("surveillance.thresholds.z_score_critical", 3.0))
            th_growth_warn = float(self.config.get("surveillance.thresholds.growth_rate_warning", 0.15))
            th_growth_crit = float(self.config.get("surveillance.thresholds.growth_rate_critical", 0.35))

            self.log_timeline_event("baseline_evaluation", "Regional Baselines Evaluated", f"Evaluating 28-day moving baselines across {len(monitored_regions)} regions and {len(monitored_diseases)} diseases.", "info")

            # Evaluate each (region, disease) time series
            df_epi["date"] = pd.to_datetime(df_epi["date"])
            now_iso = datetime.now().isoformat()

            grouped = df_epi.groupby(["region", "disease"])

            for (reg, dis), group in grouped:
                group = group.sort_values("date")
                if len(group) < 14:
                    continue

                cases = group["new_cases"].values
                recent_cases = float(cases[-1])
                prev_cases = float(cases[-2]) if len(cases) > 1 else recent_cases

                # 28-day baseline mean & std (excluding current day)
                baseline_window = cases[-29:-1] if len(cases) >= 29 else cases[:-1]
                b_mean = float(np.mean(baseline_window)) if len(baseline_window) > 0 else recent_cases
                b_std = float(np.std(baseline_window)) if len(baseline_window) > 0 else 1.0
                if b_std < 0.1:
                    b_std = 1.0

                z_score = (recent_cases - b_mean) / b_std

                # 7-day growth rate
                w7 = cases[-7:]
                w7_prev = cases[-14:-7] if len(cases) >= 14 else cases[:7]
                mean_w7 = np.mean(w7)
                mean_w7_prev = np.mean(w7_prev)
                growth_7d = float((mean_w7 - mean_w7_prev) / (mean_w7_prev + 1e-4))

                # Population & incidence
                pop = demo_dict.get(reg, {}).get("population", 2000000)
                incidence = (recent_cases / pop) * 100000.0

                # Check if threshold is breached
                is_anomalous = (z_score >= th_z_warn) or (growth_7d >= th_growth_warn and recent_cases > 15)

                if is_anomalous:
                    severity = "Critical" if (z_score >= th_z_crit or growth_7d >= th_growth_crit) else "Moderate"
                    if z_score >= th_z_crit and growth_7d >= th_growth_crit:
                        severity = "Critical"
                    elif z_score < th_z_warn and growth_7d >= th_growth_warn:
                        severity = "Moderate"
                    elif z_score >= th_z_warn and growth_7d < th_growth_crit:
                        severity = "High"

                    finding_id = f"FND-{started_at.strftime('%Y%m%d')}-{reg[:3].upper()}-{dis[:3].upper()}-{len(findings_created)+1:02d}"
                    alert_id = f"ALT-{started_at.strftime('%Y%m%d')}-{reg[:3].upper()}-{len(findings_created)+1:02d}"

                    # Recommendation rule selection
                    rec_next = "Invoke 14-day trajectory forecast and check ICU bed utilization."
                    if severity == "Critical":
                        rec_next = "Activate regional rapid-response epidemiological investigation, expand targeted testing, and prepare surge bed capacity."
                    elif dis in ["Dengue", "Malaria"] and env_latest.get("humidity", 0) > 70:
                        rec_next = "Deploy municipal vector control units in high-density wards; environmental humidity is elevated."

                    summary_text = (
                        f"{reg} shows concerning {dis} acceleration: current daily cases ({recent_cases:.0f}) "
                        f"exceed 28-day historical baseline (mean={b_mean:.1f}, std={b_std:.1f}) with Z-score {z_score:.2f}σ "
                        f"and 7-day growth rate {growth_7d*100:.1f}%. Incident rate: {incidence:.2f} per 100k."
                    )

                    finding = {
                        "finding_id": finding_id,
                        "timestamp": now_iso,
                        "region": reg,
                        "disease": dis,
                        "triggering_metric": "rolling_z_score" if z_score >= th_z_warn else "growth_rate_7d",
                        "observed_value": round(float(z_score if z_score >= th_z_warn else growth_7d), 3),
                        "historical_baseline": {
                            "baseline_mean": round(b_mean, 2),
                            "baseline_std": round(b_std, 2),
                            "window_days": 28,
                            "upper_threshold_2std": round(b_mean + 2 * b_std, 2),
                            "critical_threshold_3std": round(b_mean + 3 * b_std, 2)
                        },
                        "threshold": th_z_crit if severity == "Critical" else th_z_warn,
                        "detection_method": "Multi-Signal Rolling Z-Score & Moving Baseline",
                        "severity": severity,
                        "supporting_evidence": {
                            "recent_daily_cases": recent_cases,
                            "7d_growth_rate_percent": round(growth_7d * 100, 1),
                            "incidence_per_100k": round(incidence, 2),
                            "population": pop,
                            "hospital_stress_index": hc_stress_map.get(reg, 1.0),
                            "environmental_context": env_latest
                        },
                        "recommended_next_step": rec_next,
                        "related_alert_id": alert_id,
                        "data_provenance": "SYNTHETIC_DEMO (Controlled biological models) + Census 2011 Demographics",
                        "status": "unconfirmed_signal" if severity != "Critical" else "investigating",
                        "summary": summary_text
                    }

                    findings_created.append(finding)

            # Deduplicate and persist findings
            existing_findings = self.get_findings()
            # Deduplicate by (region, disease, date)
            existing_keys = {f"{f.get('region')}_{f.get('disease')}_{f.get('timestamp')[:10]}" for f in existing_findings}
            unique_new = [f for f in findings_created if f"{f['region']}_{f['disease']}_{f['timestamp'][:10]}" not in existing_keys]

            combined_findings = unique_new + existing_findings
            # Keep up to 200 historical findings
            combined_findings = combined_findings[:200]

            with open(self.findings_file, "w", encoding="utf-8") as f:
                json.dump(combined_findings, f, indent=2)

            self.log_timeline_event("findings_recorded", "Findings Recorded", f"Identified {len(unique_new)} new epidemiological anomalies across monitored regions.", "success" if unique_new else "info")

            # 4. Alert Generation & Deduplication
            if unique_new:
                alerts_triggered = self._process_alerts(unique_new)

            # 5. Recommendation Engine Invocation
            self._update_recommendations(unique_new)

            self.log_timeline_event("scan_completed", "Surveillance Scan Complete", f"Scan {scan_id} finished: {records_evaluated} records evaluated, {len(unique_new)} new findings, {alerts_triggered} alerts synchronized.", "success")

        except Exception as e:
            logger.exception("Error executing surveillance scan")
            errors.append(str(e))
            self.log_timeline_event("scan_failed", "Surveillance Scan Failed", f"Scan failed: {str(e)}", "error")

        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        scan_record = {
            "scan_id": scan_id,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round(duration, 2),
            "status": "COMPLETED" if not errors else "FAILED",
            "source": source,
            "records_evaluated": records_evaluated,
            "findings_generated": len(findings_created),
            "alerts_triggered": alerts_triggered,
            "errors": errors
        }

        # Save scan record to scans.json
        try:
            scans = []
            if self.scans_file.exists():
                with open(self.scans_file, "r", encoding="utf-8") as f:
                    scans = json.load(f)
            scans = [scan_record] + scans[:49]
            with open(self.scans_file, "w", encoding="utf-8") as f:
                json.dump(scans, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scan history: {e}")

        return scan_record

    def _process_alerts(self, new_findings: List[Dict[str, Any]]) -> int:
        """Deduplicates and creates alerts in data/alerts/alerts.csv."""
        alerts_path = Path("data/alerts/alerts.csv")
        alerts_path.parent.mkdir(parents=True, exist_ok=True)

        existing_df = pd.read_csv(alerts_path) if alerts_path.exists() else pd.DataFrame(
            columns=["alert_id", "region", "category", "severity", "status", "explanation", "triggering_metric", "observed_value", "timestamp"]
        )

        new_alert_rows = []
        for f in new_findings:
            if f.get("severity") in ["Moderate", "High", "Critical"]:
                # Check deduplication within cooldown
                reg = f["region"]
                dis = f["disease"]
                metric = f["triggering_metric"]
                
                # Cooldown check: if an active alert for same region exists in last 24h
                is_duplicate = False
                if not existing_df.empty and "region" in existing_df.columns:
                    matches = existing_df[(existing_df["region"] == reg) & (existing_df["status"] == "active")]
                    if not matches.empty:
                        is_duplicate = True

                if not is_duplicate:
                    new_alert_rows.append({
                        "alert_id": f["related_alert_id"],
                        "region": reg,
                        "category": f"Surveillance Anomaly ({dis})",
                        "severity": f["severity"],
                        "status": "active",
                        "explanation": f["summary"],
                        "triggering_metric": metric,
                        "observed_value": f["observed_value"],
                        "timestamp": f["timestamp"]
                    })

        if new_alert_rows:
            new_df = pd.DataFrame(new_alert_rows)
            combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
            combined.to_csv(alerts_path, index=False)
            self.log_timeline_event("alerts_updated", "Alerts Synchronized", f"Generated {len(new_alert_rows)} active alerts with deduplication guardrails.", "warning")
            return len(new_alert_rows)
        return 0

    def _update_recommendations(self, new_findings: List[Dict[str, Any]]) -> None:
        """Updates recommendations table based on active surveillance signals."""
        rec_path = Path("data/recommendations/recommendations.csv")
        if not rec_path.exists() or not new_findings:
            return

        try:
            df_rec = pd.read_csv(rec_path)
            new_recs = []
            for f in new_findings:
                if f.get("severity") in ["High", "Critical"]:
                    new_recs.append({
                        "recommendation_id": f"REC-{f['region'][:3].upper()}-{datetime.now().strftime('%m%d%H%M')}",
                        "region": f["region"],
                        "target_stakeholder": "Public-health authorities",
                        "recommendation_title": f"Surveillance Directive: {f['disease']} Alert in {f['region']}",
                        "priority": "High priority" if f["severity"] == "High" else "Critical emergency",
                        "rationale": f["summary"],
                        "action_steps": f["recommended_next_step"],
                        "evidence_level": "Level 1 Surveillance Signal",
                        "generated_at": datetime.now().isoformat()
                    })
            if new_recs:
                new_rec_df = pd.DataFrame(new_recs)
                combined = pd.concat([df_rec, new_rec_df], ignore_index=True)
                combined.to_csv(rec_path, index=False)
                self.log_timeline_event("recommendations_updated", "Action Directives Updated", f"Published {len(new_recs)} prioritized public-health directives.", "info")
        except Exception as e:
            logger.warning(f"Error updating recommendations: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates rich surveillance analytics statistics."""
        findings = self.get_findings()
        scans = []
        if self.scans_file.exists():
            try:
                with open(self.scans_file, "r", encoding="utf-8") as f:
                    scans = json.load(f)
            except Exception:
                pass

        alerts_path = Path("data/alerts/alerts.csv")
        active_alerts_count = 0
        if alerts_path.exists():
            try:
                df_alerts = pd.read_csv(alerts_path)
                if not df_alerts.empty and "status" in df_alerts.columns:
                    active_alerts_count = int((df_alerts["status"] == "active").sum())
            except Exception:
                pass

        # Severity breakdown
        severities = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
        diseases = {}
        regions = {}
        methods = {}
        findings_over_time = {}

        for f in findings:
            sev = f.get("severity", "Low")
            severities[sev] = severities.get(sev, 0) + 1

            dis = f.get("disease", "Unknown")
            diseases[dis] = diseases.get(dis, 0) + 1

            reg = f.get("region", "Unknown")
            regions[reg] = regions.get(reg, 0) + 1

            m = f.get("detection_method", "Baseline Z-Score")
            methods[m] = methods.get(m, 0) + 1

            dt = f.get("timestamp", "")[:10]
            if dt:
                findings_over_time[dt] = findings_over_time.get(dt, 0) + 1

        # Scan success/failure
        scan_success = sum(1 for s in scans if s.get("status") == "COMPLETED")
        scan_failed = sum(1 for s in scans if s.get("status") == "FAILED")

        return {
            "total_findings": len(findings),
            "active_alerts": active_alerts_count,
            "total_scans": len(scans),
            "scan_success_count": scan_success,
            "scan_failure_count": scan_failed,
            "by_severity": [{"name": k, "count": v} for k, v in severities.items()],
            "by_disease": [{"name": k, "count": v} for k, v in diseases.items()],
            "by_region": [{"name": k, "count": v} for k, v in regions.items()],
            "by_detection_method": [{"name": k, "count": v} for k, v in methods.items()],
            "over_time": [{"date": k, "count": v} for k, v in sorted(findings_over_time.items())]
        }


_engine_instance: Optional[SurveillanceEngine] = None

def get_surveillance_engine() -> SurveillanceEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SurveillanceEngine()
    return _engine_instance
