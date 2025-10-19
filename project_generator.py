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
    elif shape == "constant":
        y = np.where((x == 0.0) | (x == 1.0), 0.0, 1.0)
    else:
        y = np.where((x == 0.0) | (x == 1.0), 0.0, 1.0)
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
    flag_random = False # DEBUG
    if flag_random:
        np.random.seed(seed)

    grants = []
    grants_BL = []
    start_peaks = []  # DEBUG
    for _ in range(n_grants_baseline):
        start = _safe_randint(-36, n_months)
        dur = np.random.randint(12, 36)
        # peak = np.random.uniform(0.5, 2.5) # DEBUG
        peak_BL = 1.0 # DEBUG
        if start > decline_month:
            peak = peak_BL * decline_factor
        else:
            peak = peak_BL

        # grants.append(burn_curve(t, start, dur, peak, shape)) # DEBUG
        grants.append(burn_curve(t, start, 40, peak, shape))  # DEBUG
        grants_BL.append(burn_curve(t, start, 40, peak_BL, shape))  # DEBUG
        start_peaks.append([start,peak])  # DEBUG

    print(f"start_peaks={start_peaks}")
    # --- after decline projects ---
    # for _ in range(n_after):
    #     # valid starts in [decline_month, n_months-1]; use safe randint
    #     start = _safe_randint(decline_month, max(n_months, decline_month + 1))
    #     dur   = np.random.randint(12, 36)
    #     peak  = np.random.uniform(0.5, 2.5)
    #     grants.append(burn_curve(t, start, dur, peak, shape))

    return (np.array(grants),np.array(grants_BL)) if grants else ( np.zeros((0, len(t))), np.zeros((0, len(t))) )