"""Statistical analysis using GLM."""
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import List, Dict, Any

def fit_poisson_model(
    df: pd.DataFrame,
    target: str,
    predictors: List[str]
) -> Dict[str, Any]:
    """Fit a Poisson regression model."""
    
    # Filter available predictors and target
    available_preds = [p for p in predictors if p in df.columns]
    if not available_preds or target not in df.columns:
        return {'success': False, 'message': 'Missing predictors or target.'}
        
    # Drop NAs
    valid_df = df[[target] + available_preds].dropna()
    
    if len(valid_df) < 30:
        return {'success': False, 'message': 'Insufficient samples for statistical modeling.'}
        
    formula = f"{target} ~ " + " + ".join(available_preds)
    
    try:
        model = smf.glm(formula=formula, data=valid_df, family=sm.families.Poisson()).fit()
        return {
            'success': True,
            'model': model,
            'summary': str(model.summary()),
            'aic': model.aic,
            'deviance': model.deviance,
            'params': model.params.to_dict()
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}

def fit_negative_binomial_model(
    df: pd.DataFrame,
    target: str,
    predictors: List[str]
) -> Dict[str, Any]:
    """Fit a Negative Binomial regression model."""
    
    available_preds = [p for p in predictors if p in df.columns]
    if not available_preds or target not in df.columns:
        return {'success': False, 'message': 'Missing predictors or target.'}
        
    valid_df = df[[target] + available_preds].dropna()
    
    if len(valid_df) < 30:
        return {'success': False, 'message': 'Insufficient samples.'}
        
    formula = f"{target} ~ " + " + ".join(available_preds)
    
    try:
        # GLM with NegativeBinomial requires an alpha parameter.
        # Often it's easier to use sm.GLM with NegativeBinomial family or sm.NegativeBinomial.
        # Using GLM with alpha=1 as a starting point.
        model = smf.glm(formula=formula, data=valid_df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        return {
            'success': True,
            'model': model,
            'summary': str(model.summary()),
            'aic': model.aic,
            'deviance': model.deviance,
            'params': model.params.to_dict()
        }
    except Exception as e:
        return {'success': False, 'message': str(e)}
