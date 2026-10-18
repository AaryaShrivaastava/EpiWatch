"""Risk explanation."""

def generate_risk_explanation(
    region: str,
    category: str,
    score: float,
    assessments: dict,
    contributions: dict
) -> str:
    """Generate textual explanation from contributing factors."""
    
    if category == "Insufficient Data":
        return f"Regional assessment: Insufficient Data. Essential inputs are missing for {region}."
        
    text = f"Regional assessment: {category}. "
    
    # Identify top contributors
    if contributions:
        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_factors = [k.replace('_', ' ') for k, v in sorted_contribs[:2] if v > 10.0]
        
        if top_factors:
            text += f"The classification is primarily associated with elevated {', and '.join(top_factors)}. "
            
    # Check forecast
    fc = assessments.get('forecast_trajectory', {})
    if fc.get('status') == 'available':
        if fc['score'] > 50:
            text += "The forecast suggests further growth, but uncertainty remains. "
        else:
            text += "The forecast suggests stable or declining burden. "
            
    # Check missing
    missing = [k.replace('_', ' ') for k, v in assessments.items() if v['status'] != 'available']
    if missing:
        text += f"{', '.join(missing).capitalize()} evidence is unavailable for this period, so the integrated assessment has reduced coverage. "
        
    text += "This assessment is model-based and does not confirm a clinical outbreak."
    
    return text
