#!/usr/bin/env python3
"""
Calculate CV fold thresholds for different data sizes
"""


def calculate_fold_thresholds():
    """Calculate data size thresholds for different fold counts"""

    print("=== CV FOLD THRESHOLDS ANALYSIS ===\n")

    # Test a range of data sizes to find thresholds
    data_sizes = range(20, 2001, 20)  # 20 to 2000 days, step by 20

    fold_boundaries = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

    for data_size in data_sizes:
        # Simulate your algorithm for different scenarios

        # Scenario 1: Small data (< 2 years = 504 trading days)
        if data_size < 504:
            period_unit = data_size // 4
            train_period = data_size
        else:
            # Scenario 2: Large data (>= 2 years)
            period_unit = 365  # Calendar days (your current setting)
            train_period = data_size

        # Your CV algorithm
        cv_initial = min(train_period, int(data_size * 0.5))
        max_reasonable_period = int((data_size - cv_initial) / 2)
        cv_period = max(
            min(period_unit, max_reasonable_period),
            int((data_size - cv_initial) / 3),
        )

        # Calculate folds
        remaining_data = data_size - cv_initial
        folds = remaining_data // cv_period if cv_period > 0 else 0

        # Cap at 5 folds for analysis
        folds = min(folds, 5)

        fold_boundaries[folds].append(data_size)

    # Find threshold boundaries
    print("FOLD COUNT THRESHOLDS:")
    print("=" * 50)

    for fold_count in sorted(fold_boundaries.keys()):
        sizes = fold_boundaries[fold_count]
        if sizes:
            min_size = min(sizes)
            max_size = max(sizes)
            print(f"{fold_count} folds: {min_size:4d} - {max_size:4d} days")
        else:
            print(f"{fold_count} folds: No data sizes produce this")

    print("\n" + "=" * 50)

    # Key thresholds
    print("\nKEY THRESHOLDS:")
    print("=" * 30)

    # Find where 0 folds ends and 1+ folds begins
    zero_fold_sizes = fold_boundaries[0]
    one_plus_fold_sizes = []
    for i in range(1, 6):
        one_plus_fold_sizes.extend(fold_boundaries[i])

    if zero_fold_sizes and one_plus_fold_sizes:
        print(f"0 folds → 1+ folds: ~{max(zero_fold_sizes)} days")

    # Find where 1 fold ends and 2+ folds begins
    one_fold_sizes = fold_boundaries[1]
    two_plus_fold_sizes = []
    for i in range(2, 6):
        two_plus_fold_sizes.extend(fold_boundaries[i])

    if one_fold_sizes and two_plus_fold_sizes:
        print(f"1 fold → 2+ folds: ~{max(one_fold_sizes)} days")

    # Trading day equivalents
    print("\nTRADING DAY EQUIVALENTS:")
    print("=" * 40)
    print("(Assuming ~252 trading days per year)")

    key_thresholds = [60, 120, 250, 500, 750, 1000, 1500, 2000]
    for threshold in key_thresholds:
        trading_years = threshold / 252

        # Calculate folds for this threshold
        if threshold < 504:
            period_unit = threshold // 4
            train_period = threshold
        else:
            period_unit = 365
            train_period = threshold

        cv_initial = min(train_period, int(threshold * 0.5))
        max_reasonable_period = int((threshold - cv_initial) / 2)
        cv_period = max(
            min(period_unit, max_reasonable_period),
            int((threshold - cv_initial) / 3),
        )

        remaining_data = threshold - cv_initial
        folds = remaining_data // cv_period if cv_period > 0 else 0

        print(f"{threshold:4d} days ({trading_years:4.1f} years): {folds} folds")


if __name__ == "__main__":
    calculate_fold_thresholds()
