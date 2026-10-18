"""Reproduction number estimation module."""

def calculate_r0_sir(beta: float, gamma: float) -> float:
    """
    Calculate the basic reproduction number (R0) for the simplified SIR model.
    Assumptions: Random mixing, homogeneous population, no demographic changes.
    """
    if gamma <= 0:
        return float('inf')
    return beta / gamma

def calculate_r0_sird(beta: float, gamma: float, mu: float) -> float:
    """
    Calculate the basic reproduction number (R0) for the simplified SIRD model.
    Assumptions: Recovery and death both remove infectious individuals.
    """
    if (gamma + mu) <= 0:
        return float('inf')
    return beta / (gamma + mu)

def estimate_rt_naive(beta_t: float, S_t: float, N: float, gamma: float) -> float:
    """
    Calculate the effective reproduction number Rt at time t.
    This is a naive estimate: Rt = R0 * (S_t / N)
    For actual Rt estimation from incidence curves, more advanced methods (like EpiEstim)
    are usually required.
    """
    if gamma <= 0:
        return float('inf')
    return (beta_t / gamma) * (S_t / N)
