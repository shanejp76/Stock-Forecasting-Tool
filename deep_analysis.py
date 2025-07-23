#!/usr/bin/env python3
"""
Deep analysis of why we're getting exactly 2 folds always
"""


def analyze_algorithm_behavior():
    """Analyze why the algorithm always produces 2 folds"""

    print("=== DETAILED ALGORITHM ANALYSIS ===\n")

    test_cases = [100, 250, 500, 1000, 2000]

    for data_size in test_cases:
        print(f"=== {data_size} days ===")

        # Determine period_unit based on data size
        if data_size < 504:  # < 2 years
            period_unit = data_size // 4
            train_period = data_size
            scenario = "Small data"
        else:  # >= 2 years
            period_unit = 365
            train_period = data_size
            scenario = "Large data"

        # Your algorithm step by step
        cv_initial = min(train_period, int(data_size * 0.5))
        max_reasonable_period = int((data_size - cv_initial) / 2)

        option_1 = period_unit
        option_2 = max_reasonable_period
        option_3 = int((data_size - cv_initial) / 3)

        cv_period = max(
            min(option_1, option_2),  # min(period_unit, max_reasonable_period)
            option_3,  # int((available_data - cv_initial) / 3)
        )

        remaining_data = data_size - cv_initial
        folds = remaining_data // cv_period if cv_period > 0 else 0

        print(f"Scenario: {scenario}")
        print(f"Initial: {cv_initial} ({cv_initial/data_size:.1%} of data)")
        print(f"Remaining: {remaining_data}")
        print(f"")
        print(f"Option 1 (period_unit): {option_1}")
        print(f"Option 2 (max_reasonable): {option_2}")
        print(f"Option 3 (remaining/3): {option_3}")
        print(f"")
        print(f"min(option_1, option_2) = {min(option_1, option_2)}")
        print(f"max(min_result, option_3) = {cv_period}")
        print(f"")
        print(f"Final CV period: {cv_period}")
        print(f"Folds: {remaining_data} ÷ {cv_period} = {folds}")
        print(f"{'='*40}\n")


def find_fold_boundaries():
    """Find exact boundaries where fold count changes"""

    print("=== FINDING EXACT FOLD BOUNDARIES ===\n")

    # The key insight: max_reasonable_period always equals remaining_data / 2
    # So we get exactly 2 folds when max_reasonable_period is chosen

    print("KEY INSIGHT:")
    print("max_reasonable_period = (data_size - cv_initial) / 2")
    print("When this is chosen as cv_period:")
    print(
        "folds = remaining_data / cv_period = remaining_data / (remaining_data/2) = 2"
    )
    print("")

    print("The algorithm will choose max_reasonable_period when:")
    print(
        "max_reasonable_period >= remaining_data/3  AND  max_reasonable_period <= period_unit"
    )
    print("")

    # Test when we might get different fold counts
    print("SCENARIOS FOR DIFFERENT FOLD COUNTS:")
    print("=" * 50)

    for data_size in [50, 100, 200, 400, 600, 1000]:
        if data_size < 504:
            period_unit = data_size // 4
        else:
            period_unit = 365

        cv_initial = int(data_size * 0.5)
        remaining = data_size - cv_initial
        max_reasonable = remaining // 2
        option_3 = remaining // 3

        print(f"{data_size} days:")
        print(f"  period_unit: {period_unit}")
        print(f"  max_reasonable: {max_reasonable}")
        print(f"  remaining/3: {option_3}")

        if max_reasonable <= period_unit and max_reasonable >= option_3:
            chosen = max_reasonable
            folds = 2
            print(f"  → Chooses max_reasonable = {chosen} → {folds} folds ✓")
        elif period_unit < max_reasonable:
            chosen = max(period_unit, option_3)
            folds = remaining // chosen
            print(
                f"  → Chooses max({period_unit}, {option_3}) = {chosen} → {folds} folds"
            )
        else:
            chosen = option_3
            folds = 3
            print(f"  → Chooses remaining/3 = {chosen} → {folds} folds")
        print()


if __name__ == "__main__":
    analyze_algorithm_behavior()
    find_fold_boundaries()
