# Function to calculate different Kelly Criterion approaches

def kelly_criterion(odds, win_prob, capital, risk_adjustment=0, fractional_kelly=1, cap_fraction=1):
    """
    Calculate the amount to wager using Risk-Adjusted Kelly and Capped Kelly formulas.

    Parameters:
        odds (float): Decimal odds of the bet (e.g., 2.5 for 3:2 odds).
        win_prob (float): Probability of winning the bet.
        capital (float): Total available capital.
        risk_adjustment (float): Adjustment for uncertainty in win probability (0 by default).
        fractional_kelly (float): Fraction to scale the Kelly fraction (1 = full Kelly, <1 = fractional Kelly).
        cap_fraction (float): Maximum fraction of capital allowed to be wagered.

    Returns:
        dict: A dictionary containing wager amounts and percentages for different Kelly approaches.
    """
    # Adjust win probability to account for risk
    adjusted_win_prob = max(0, win_prob - risk_adjustment)

    # Calculate full Kelly fraction
    b = odds - 1
    full_kelly_fraction = (b * win_prob - (1 - win_prob)) / b
    full_kelly_fraction = max(0, full_kelly_fraction)  # Ensure non-negative

    # Calculate risk-adjusted Kelly fraction
    risk_adjusted_fraction = (b * adjusted_win_prob - (1 - adjusted_win_prob)) / b
    risk_adjusted_fraction = max(0, risk_adjusted_fraction)  # Ensure non-negative

    # Apply fractional scaling
    fractional_kelly_fraction = full_kelly_fraction * fractional_kelly
    half_kelly_fraction = full_kelly_fraction * 0.5
    quarter_kelly_fraction = full_kelly_fraction * 0.25

    # Apply cap to Kelly fractions
    capped_full_kelly_fraction = min(full_kelly_fraction, cap_fraction)
    capped_risk_adjusted_fraction = min(risk_adjusted_fraction, cap_fraction)
    capped_half_kelly_fraction = min(half_kelly_fraction, cap_fraction)
    capped_quarter_kelly_fraction = min(quarter_kelly_fraction, cap_fraction)

    # Calculate wager amounts and percentages
    wagers = {
        #"Full Kelly": (full_kelly_fraction * capital, full_kelly_fraction * 100),
        #"Risk-Adjusted Kelly": (risk_adjusted_fraction * capital, risk_adjusted_fraction * 100),
       # "Fractional Kelly": (fractional_kelly_fraction * capital, fractional_kelly_fraction * 100),
        #"Half Kelly": (half_kelly_fraction * capital, half_kelly_fraction * 100),
       # "Quarter Kelly": (quarter_kelly_fraction * capital, quarter_kelly_fraction * 100),
        "Capped Full Kelly": (capped_full_kelly_fraction * capital, capped_full_kelly_fraction * 100),
        "Capped Risk-Adjusted Kelly": (capped_risk_adjusted_fraction * capital, capped_risk_adjusted_fraction * 100),
        "Capped Half Kelly": (capped_half_kelly_fraction * capital, capped_half_kelly_fraction * 100),
        "Capped Quarter Kelly": (capped_quarter_kelly_fraction * capital, capped_quarter_kelly_fraction * 100),
    }

    return wagers

# Example Parameters
odds = 1.77      # Decimal odds (e.g., 2.5 means 3:2 odds)
win_prob = 0.7    # Estimated probability of winning
capital = 35.12       # Total available capital
risk_adjustment = 0.05    # Adjust win probability by 5% for risk
fractional_kelly = 0.5    # Use 50% of full Kelly fraction
cap_fraction = 0.25       # Cap wagers at 10% of capital

# Calculate wagers
wagers = kelly_criterion(odds, win_prob, capital, risk_adjustment, fractional_kelly, cap_fraction)

# Display results
for kelly_type, (wager, percentage) in wagers.items():
    print(f"{kelly_type}: ${wager:.2f} ({percentage:.2f}%)")
