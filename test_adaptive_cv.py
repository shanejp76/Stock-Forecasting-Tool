#!/usr/bin/env python3
"""
Test the new adaptive CV folds algorithm
"""


def test_new_adaptive_algorithm():
    """Test the improved adaptive CV algorithm"""

    print("=== NEW ADAPTIVE CV ALGORITHM TEST ===\n")

    def calculate_cv_folds(
        available_data, train_period_base, period_unit_base, forecast_period_base
    ):
        # New adaptive algorithm
        cv_initial = min(train_period_base, int(available_data * 0.5))

        # Adaptive CV folds based on data size
        remaining_data = available_data - cv_initial

        # Determine target folds based on data size
        if available_data <= 250:  # ~1 year or less
            target_folds = 2
        elif available_data <= 500:  # ~2 years
            target_folds = 3
        elif available_data <= 1000:  # ~4 years
            target_folds = 4
        else:  # > 4 years
            target_folds = 5

        # Calculate period to achieve target folds, but respect limits
        target_period = remaining_data // target_folds
        min_period = 30  # Minimum 1 month spacing
        max_period = min(
            period_unit_base, remaining_data // 2
        )  # Cap by period_unit and ensure at least 2 folds

        cv_period = max(min_period, min(target_period, max_period))

        cv_horizon = min(forecast_period_base, 30, int(available_data * 0.1))

        # Calculate actual folds
        actual_folds = remaining_data // cv_period if cv_period > 0 else 0

        return cv_initial, cv_period, cv_horizon, actual_folds, target_folds

    # Test scenarios
    test_cases = [
        {
            "data": 100,
            "period_unit": 25,
            "forecast": 25,
            "train": 100,
            "desc": "100 days (~3 months)",
        },
        {
            "data": 250,
            "period_unit": 63,
            "forecast": 63,
            "train": 250,
            "desc": "250 days (~1 year)",
        },
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
        {
            "data": 1000,
            "period_unit": 365,
            "forecast": 365,
            "train": 1000,
            "desc": "1000 days (~4 years)",
        },
        {
            "data": 1500,
            "period_unit": 365,
            "forecast": 365,
            "train": 1500,
            "desc": "1500 days (~6 years)",
        },
        {
            "data": 2000,
            "period_unit": 365,
            "forecast": 365,
            "train": 2000,
            "desc": "2000 days (~8 years)",
        },
    ]

    print("DATA SIZE → TARGET FOLDS → ACTUAL FOLDS")
    print("=" * 50)

    for case in test_cases:
        initial, period, horizon, actual_folds, target_folds = calculate_cv_folds(
            case["data"], case["train"], case["period_unit"], case["forecast"]
        )

        remaining = case["data"] - initial

        # Status indicator
        if actual_folds >= target_folds:
            status = "✅"
        elif actual_folds >= 2:
            status = "⚠️ "
        else:
            status = "❌"

        print(f"{status} {case['desc']}")
        print(f"    Target: {target_folds} folds → Actual: {actual_folds} folds")
        print(f"    CV: initial={initial}, period={period}, horizon={horizon}")
        print(f"    Remaining data: {remaining}, Period: {period}")
        print()


def compare_old_vs_new():
    """Compare old vs new algorithms"""

    print("\n=== OLD vs NEW COMPARISON ===\n")

    def old_algorithm(available_data, period_unit):
        cv_initial = int(available_data * 0.5)
        remaining = available_data - cv_initial
        max_reasonable = remaining // 2
        option_3 = remaining // 3
        cv_period = max(min(period_unit, max_reasonable), option_3)
        return remaining // cv_period if cv_period > 0 else 0

    def new_algorithm(available_data):
        cv_initial = int(available_data * 0.5)
        remaining = available_data - cv_initial

        if available_data <= 250:
            target_folds = 2
        elif available_data <= 500:
            target_folds = 3
        elif available_data <= 1000:
            target_folds = 4
        else:
            target_folds = 5

        target_period = remaining // target_folds
        cv_period = max(30, target_period)  # Simplified for comparison
        return remaining // cv_period if cv_period > 0 else 0

    test_sizes = [250, 500, 750, 1000, 1500, 2000]

    print("DATA SIZE | OLD FOLDS | NEW FOLDS | IMPROVEMENT")
    print("-" * 50)

    for size in test_sizes:
        period_unit = 365 if size >= 504 else size // 4
        old_folds = old_algorithm(size, period_unit)
        new_folds = new_algorithm(size)
        improvement = new_folds - old_folds

        print(f"{size:8d} | {old_folds:9d} | {new_folds:9d} | {improvement:+11d}")


if __name__ == "__main__":
    test_new_adaptive_algorithm()
    compare_old_vs_new()
