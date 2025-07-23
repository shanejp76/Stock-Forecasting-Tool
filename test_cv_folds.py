#!/usr/bin/env python3
"""
Quick test script to verify CV folds calculation
"""


def test_cv_folds(
    available_data, train_period_base, period_unit_base, forecast_period_base
):
    """Test the improved CV fold algorithm"""
    print(f"\n=== Testing {available_data} rows of data ===")

    # Simulate the improved algorithm from main.py
    cv_initial = min(train_period_base, int(available_data * 0.5))

    # Improved CV period calculation
    max_reasonable_period = int((available_data - cv_initial) / 2)  # At least 2 folds
    cv_period = max(
        min(period_unit_base, max_reasonable_period),
        int((available_data - cv_initial) / 3),
    )

    cv_horizon = min(forecast_period_base, 30, int(available_data * 0.1))

    print(f"CV Initial: {cv_initial} days (training window)")
    print(f"CV Period: {cv_period} days (fold spacing) - was {period_unit_base}")
    print(f"CV Horizon: {cv_horizon} days (forecast window)")

    # Calculate approximate number of folds
    remaining_data = available_data - cv_initial
    if cv_period > 0:
        approx_folds = remaining_data // cv_period
    else:
        approx_folds = 0

    print(f"Remaining data after initial: {remaining_data} days")
    print(f"Approximate number of folds: {approx_folds}")

    # Check if CV is feasible
    min_required = cv_initial + cv_horizon
    feasible = available_data >= min_required
    print(f"CV feasible? {feasible} (needs at least {min_required} days)")

    # Status
    if approx_folds >= 2:
        status = "✅ GOOD - Proper CV validation"
    elif approx_folds == 1:
        status = "⚠️  MARGINAL - Limited validation"
    else:
        status = "❌ BAD - No cross-validation"

    print(f"Status: {status}")
    return approx_folds


def simulate_data_scenarios():
    """Simulate different data size scenarios"""
    print("=== CV FOLDS TEST - IMPROVED ALGORITHM ===")

    scenarios = [
        # Small dataset (< 2 years)
        {
            "data": 30,
            "period_unit": 7,
            "forecast": 7,
            "train": 30,
            "desc": "30 days (1 month)",
        },
        {
            "data": 100,
            "period_unit": 25,
            "forecast": 25,
            "train": 100,
            "desc": "100 days (~3 months)",
        },
        {
            "data": 252,
            "period_unit": 63,
            "forecast": 63,
            "train": 252,
            "desc": "252 days (1 trading year)",
        },
        # Medium dataset (problematic before)
        {
            "data": 500,
            "period_unit": 365,
            "forecast": 365,
            "train": 500,
            "desc": "500 days (~2 years)",
        },
        {
            "data": 750,
            "period_unit": 365,
            "forecast": 365,
            "train": 750,
            "desc": "750 days (~3 years)",
        },
        # Large dataset
        {
            "data": 1000,
            "period_unit": 365,
            "forecast": 365,
            "train": 1000,
            "desc": "1000 days (~4 years)",
        },
        {
            "data": 2000,
            "period_unit": 365,
            "forecast": 365,
            "train": 2000,
            "desc": "2000 days (~8 years)",
        },
    ]

    results = []
    for scenario in scenarios:
        folds = test_cv_folds(
            scenario["data"],
            scenario["train"],
            scenario["period_unit"],
            scenario["forecast"],
        )
        results.append((scenario["desc"], folds))

    print("\n=== SUMMARY ===")
    for desc, folds in results:
        if folds >= 2:
            status = "✅"
        elif folds == 1:
            status = "⚠️ "
        else:
            status = "❌"
        print(f"{status} {desc}: {folds} folds")


if __name__ == "__main__":
    simulate_data_scenarios()
