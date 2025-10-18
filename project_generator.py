# project_generator.py
import numpy as np

def burn_curve(t, start, dur, peak, shape="bell"):
    x = np.clip((t - start) / dur, 0, 1)
    if shape == "bell":
        y = 4 * x * (1 - x)
    elif shape == "early":
        y = np.power(1 - (1 - x)**3, 1.5) * (1 - x)
    elif shape == "late":
        y = np.power(x, 1.5) * (1 - x)**0.3
    else:
        y = x
    return peak * y

def _safe_randint(low, high_exclusive):
    """Return a random int in [low, high_exclusive); if invalid, return low."""
    if high_exclusive <= low:
        return low
    return np.random.randint(low, high_exclusive)

def generate_projects(t, n_grants_baseline, decline_month, decline_factor,
                      n_months, seed=3, shape="bell"):
    """
    Generates overlapping project burn curves before and after a decline.
    Safely handles edge cases where decline_month is 0 or >= n_months.
    """
    np.random.seed(seed)
    grants = []

    # --- counts before/after decline, with guards ---
    n_before = n_grants_baseline if decline_month > 0 else 0
    n_after  = int(n_grants_baseline * decline_factor) if decline_month < n_months else 0

    # --- before decline projects ---
    for _ in range(n_before):
        # valid starts in [0, decline_month-1]; use safe randint
        start = _safe_randint(0, max(decline_month, 1))
        dur   = np.random.randint(12, 36)
        peak  = np.random.uniform(0.5, 2.5)
        grants.append(burn_curve(t, start, dur, peak, shape))

    # --- after decline projects ---
    for _ in range(n_after):
        # valid starts in [decline_month, n_months-1]; use safe randint
        start = _safe_randint(decline_month, max(n_months, decline_month + 1))
        dur   = np.random.randint(12, 36)
        peak  = np.random.uniform(0.5, 2.5)
        grants.append(burn_curve(t, start, dur, peak, shape))

    return np.array(grants) if grants else np.zeros((0, len(t)))