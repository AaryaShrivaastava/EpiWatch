"""Rule engine for generating recommendations."""
from typing import Dict, Any, List

def evaluate_rules(evidence: Dict[str, Any], thresholds: Dict[str, Any], region: str) -> List[Dict[str, Any]]:
    """Apply business rules to generate recommendations."""
    recommendations = []
    
    # 1. Outbreak Monitoring
    if 'outbreak_anomaly_score' in evidence:
        th = thresholds.get('outbreak_monitoring', {}).get('thresholds', {}).get('anomaly_score_min', 2.0)
        if evidence['outbreak_anomaly_score'] >= th:
            recommendations.append({
                'rec_id': 'REC_OUTBREAK_001',
                'category': 'Outbreak Monitoring',
                'title': 'Verify Outbreak Anomaly Signal',
                'description': 'Review the regional disease surveillance signal. The outbreak detection module reports an anomaly.',
                'priority': 'High priority',
                'stakeholder': 'Public-health authorities',
                'rationale': f"Anomaly score {evidence['outbreak_anomaly_score']:.2f} exceeds threshold {th}.",
                'evidence': f"Anomaly score: {evidence['outbreak_anomaly_score']:.2f}",
                'limitation': 'An anomaly is not confirmation of an outbreak. Requires verification.'
            })
            
    # 2. Disease Trend
    if 'forecast_growth_rate' in evidence:
        th = thresholds.get('disease_trend', {}).get('thresholds', {}).get('forecast_growth_min', 0.2)
        if evidence['forecast_growth_rate'] >= th:
            recommendations.append({
                'rec_id': 'REC_TREND_001',
                'category': 'Disease Trend Forecasting',
                'title': 'Prepare for Projected Disease Growth',
                'description': 'Validate the latest observations against the prediction. Prepare contingency assessments if the trend persists.',
                'priority': 'High priority',
                'stakeholder': 'Public-health authorities',
                'rationale': f"Forecasted growth rate {evidence['forecast_growth_rate']:.2f} exceeds {th}.",
                'evidence': f"Forecast growth rate: {evidence['forecast_growth_rate']:.2f}",
                'limitation': 'Uncertain forecasts are not guaranteed outcomes.'
            })
            
    # 3. Environmental Risk
    if 'environmental_risk_score' in evidence:
        th = thresholds.get('environmental_risk', {}).get('thresholds', {}).get('risk_score_min', 60.0)
        if evidence['environmental_risk_score'] >= th:
            recommendations.append({
                'rec_id': 'REC_ENV_001',
                'category': 'Environmental Risk',
                'title': 'Monitor Environmental Risk Associations',
                'description': 'Examine whether the environmental pattern coincides with observed disease trends.',
                'priority': 'Medium priority',
                'stakeholder': 'Researchers and analysts',
                'rationale': f"Environmental score {evidence['environmental_risk_score']:.2f} exceeds {th}.",
                'evidence': f"Env score: {evidence['environmental_risk_score']:.2f}",
                'limitation': 'Correlation does not establish causality.'
            })
            
    # 4. Healthcare Stress
    if 'healthcare_stress_score' in evidence:
        th = thresholds.get('healthcare_capacity', {}).get('thresholds', {}).get('stress_score_min', 50.0)
        if evidence['healthcare_stress_score'] >= th:
            recommendations.append({
                'rec_id': 'REC_HC_STRESS_001',
                'category': 'Healthcare Capacity',
                'title': 'Verify Healthcare Capacity Utilization',
                'description': 'Verify current capacity and utilization figures with the relevant facility. Evaluate contingency planning.',
                'priority': 'High priority',
                'stakeholder': 'Healthcare administrators',
                'rationale': f"Stress score {evidence['healthcare_stress_score']:.2f} exceeds {th}.",
                'evidence': f"Stress score: {evidence['healthcare_stress_score']:.2f}",
                'limitation': 'Requires local administrative validation.'
            })
            
    # 5. Healthcare Preparedness
    if 'healthcare_preparedness_score' in evidence:
        th = thresholds.get('healthcare_preparedness', {}).get('thresholds', {}).get('prep_score_max', 50.0)
        if evidence['healthcare_preparedness_score'] <= th:
            recommendations.append({
                'rec_id': 'REC_HC_PREP_001',
                'category': 'Healthcare Preparedness',
                'title': 'Review Structural Healthcare Preparedness',
                'description': 'Validate staffing and resource inventories. Review indicators with the responsible authority.',
                'priority': 'Medium priority',
                'stakeholder': 'Healthcare administrators',
                'rationale': f"Preparedness score {evidence['healthcare_preparedness_score']:.2f} is below optimal {th}.",
                'evidence': f"Prep score: {evidence['healthcare_preparedness_score']:.2f}",
                'limitation': 'A low score does not strictly prove a facility is unsafe.'
            })
            
    # 6. Resource Gaps
    if 'hospital_bed_gap' in evidence and evidence['hospital_bed_gap'] < 0:
        recommendations.append({
            'rec_id': 'REC_HC_GAP_001',
            'category': 'Healthcare Capacity',
            'title': 'Address Projected Resource Shortfall',
            'description': 'Conduct a resource-gap assessment. Escalate capacity concerns to authority.',
            'priority': 'Critical review',
            'stakeholder': 'Healthcare administrators',
            'rationale': f"Projected bed shortfall detected: {evidence['hospital_bed_gap']}.",
            'evidence': f"Bed gap: {evidence['hospital_bed_gap']}",
            'limitation': 'Model-based demand estimations require clinical validation.'
        })

    for r in recommendations:
        r['region'] = region
        r['status'] = 'generated'
        
    return recommendations
