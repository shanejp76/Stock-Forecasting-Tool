#!/usr/bin/env python3
"""
Comparison test: Old vs New CV algorithm
"""


def old_cv_algorithm(
    available_data, train_period_base, period_unit_base, forecast_period_base
):
    """Original algorithm before the fix"""
    cv_initial = min(train_period_base, int(available_data * 0.5))

    # OLD algorithm - no capping of period_unit
    cv_period = max(period_unit_base, int((available_data - cv_initial) / 3))

    cv_horizon = min(forecast_period_base, 30, int(available_data * 0.1))

    remaining_data = available_data - cv_initial
    approx_folds = remaining_data // cv_period if cv_period > 0 else 0

    return cv_initial, cv_period, cv_horizon, approx_folds


def new_cv_algorithm(
    available_data, train_period_base, period_unit_base, forecast_period_base
):
    """Improved algorithm with period capping"""
    cv_initial = min(train_period_base, int(available_data * 0.5))

    # NEW algorithm - cap period_unit for better folds
    max_reasonable_period = int((available_data - cv_initial) / 2)
    cv_period = max(
        min(period_unit_base, max_reasonable_period),
        int((available_data - cv_initial) / 3),
    )

    cv_horizon = min(forecast_period_base, 30, int(available_data * 0.1))

    remaining_data = available_data - cv_initial
    approx_folds = remaining_data // cv_period if cv_period > 0 else 0

    return cv_initial, cv_period, cv_horizon, approx_folds


def compare_algorithms():
    """Compare old vs new algorithms"""
    print("=== OLD vs NEW CV ALGORITHM COMPARISON ===\n")

    # Test problematic scenarios
    test_cases = [
        {
            "data": 500,
            "period_unit": 365,
            "forecast": 365,
            "train": 500,
            "desc": "500 days",
        },
        {
            "data": 750,
            "period_unit": 365,
            "forecast": 365,
            "train": 750,
            "desc": "750 days",
        },
        {
            "data": 1000,
            "period_unit": 365,
            "forecast": 365,
            "train": 1000,
            "desc": "1000 days",
        },
    ]

    for case in test_cases:
        print(f"=== {case['desc']} ===")

        # Old algorithm
        old_initial, old_period, old_horizon, old_folds = old_cv_algorithm(
            case["data"], case["train"], case["period_unit"], case["forecast"]
        )

        # New algorithm
        new_initial, new_period, new_horizon, new_folds = new_cv_algorithm(
            case["data"], case["train"], case["period_unit"], case["forecast"]
        )

        print(
            f"OLD: period={old_period}, folds={old_folds} {'❌' if old_folds < 2 else '✅'}"
        )
        print(
            f"NEW: period={new_period}, folds={new_folds} {'❌' if new_folds < 2 else '✅'}"
        )
        print(f"Improvement: {new_folds - old_folds} more folds\n")


if __name__ == "__main__":
    compare_algorithms()
