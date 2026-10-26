"""Menu-driven command-line application for ``SalesDataAnalyzer``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from sales_data_analyzer import SalesDataAnalyzer


def _read(prompt: str, input_func: Callable[[str], str] = input) -> str | None:
    """Read input while treating EOF and Ctrl+C as a clean cancellation."""

    try:
        return input_func(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput ended. Returning to the main menu.")
        return None


def _choose_column_with_input(
    analyzer: SalesDataAnalyzer,
    prompt: str,
    input_func: Callable[[str], str],
    allow_blank: bool = False,
) -> str | None:
    value = _read(prompt, input_func)
    if value is None or (allow_blank and not value):
        return value
    if value not in analyzer.data.columns:
        print(f"Unknown column. Available columns: {', '.join(analyzer.data.columns)}")
        return None
    return value


def _print_frame(value: Any) -> None:
    if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
        print(value.to_string())
    else:
        print(value)


def load_dataset(analyzer: SalesDataAnalyzer, input_func: Callable[[str], str] = input) -> None:
    path = _read("Enter the path of the dataset (CSV file): ", input_func)
    if not path:
        return
    try:
        analyzer.load_data(path)
    except (OSError, ValueError) as exc:
        print(f"Unable to load dataset: {exc}")
    else:
        print(f"Dataset loaded successfully: {len(analyzer.data)} rows, {len(analyzer.data.columns)} columns.")


def explore_data(analyzer: SalesDataAnalyzer, input_func: Callable[[str], str] = input) -> None:
    try:
        summary = analyzer.explore_data()
    except ValueError as exc:
        print(exc)
        return
    print("\n== Explore Data ==")
    print("1. Display the first 5 rows")
    print("2. Display the last 5 rows")
    print("3. Display column names")
    print("4. Display data types")
    print("5. Display basic info")
    print("6. Display data-quality report")
    choice = _read("Enter your choice: ", input_func)
    if choice == "1":
        _print_frame(summary["head"])
    elif choice == "2":
        _print_frame(summary["tail"])
    elif choice == "3":
        print(summary["columns"])
    elif choice == "4":
        print(pd.Series(summary["dtypes"]).to_string())
    elif choice == "5":
        print(summary["info"])
        print("Missing values:")
        _print_frame(summary["missing"])
    elif choice == "6":
        _print_frame(analyzer.data_quality_report())
    elif choice is not None:
        print("Invalid explore choice.")


def dataframe_operations(analyzer: SalesDataAnalyzer, input_func: Callable[[str], str] = input) -> None:
    if analyzer.data.empty and len(analyzer.data.columns) == 0:
        print("Load a dataset before performing DataFrame operations.")
        return
    print("\n== Perform DataFrame Operations ==")
    print("1. NumPy array operations")
    print("2. Mathematical operations")
    print("3. Search, sort, and filter")
    print("4. Split by a column")
    print("5. Create a pivot table")
    print("6. Group and aggregate")
    print("7. Export the current result")
    choice = _read("Enter your choice: ", input_func)
    try:
        if choice == "1":
            column = _choose_column_with_input(analyzer, "Numeric column (blank for first numeric column): ", input_func, allow_blank=True)
            if column is None:
                return
            result = analyzer.numpy_operations(column or None)
            print("Array:", result["array"])
            print("First value:", result["first"])
            print("First slice:", result["slice"])
            print("Doubled:", result["doubled"])
        elif choice == "2":
            left = _choose_column_with_input(analyzer, "Left numeric column: ", input_func)
            right = _choose_column_with_input(analyzer, "Right numeric column: ", input_func)
            if left is None or right is None:
                return
            operation = _read("Operation (add/subtract/multiply/divide): ", input_func)
            if operation is None:
                return
            result = analyzer.mathematical_operations(left, right, operation or "add")
            print(result.to_string())
        elif choice == "3":
            column = _choose_column_with_input(analyzer, "Search column (blank to skip): ", input_func, allow_blank=True)
            if column is None:
                return
            value = _read("Search value (blank to skip): ", input_func) if column else None
            if column and value is None:
                return
            sort_by = _choose_column_with_input(analyzer, "Sort column (blank to skip): ", input_func, allow_blank=True)
            if sort_by is None:
                return
            ascending_choice = _read("Ascending? [Y/n]: ", input_func)
            if ascending_choice is None:
                return
            ascending = (ascending_choice or "y").lower() != "n"
            result = analyzer.search_sort_filter(column or None, value, sort_by or None, ascending)
            _print_frame(result)
        elif choice == "4":
            column = _choose_column_with_input(analyzer, "Column to split by: ", input_func)
            if column:
                result = analyzer.split_data(column)
                for key, frame in result.items():
                    print(f"\n--- {key} ({len(frame)} rows) ---")
                    print(frame.to_string(index=False))
        elif choice == "5":
            index = _choose_column_with_input(analyzer, "Pivot index column: ", input_func)
            columns = _choose_column_with_input(analyzer, "Pivot columns (blank for none): ", input_func, allow_blank=True)
            values = _choose_column_with_input(analyzer, "Pivot values (blank for automatic): ", input_func, allow_blank=True)
            if index is None or columns is None or values is None:
                return
            if index:
                _print_frame(analyzer.create_pivot_table(index, columns or None, values or None))
        elif choice == "6":
            group = _choose_column_with_input(analyzer, "Group-by column (blank for overall stats): ", input_func, allow_blank=True)
            value = _choose_column_with_input(analyzer, "Value column (blank for automatic): ", input_func, allow_blank=True)
            if group is None or value is None:
                return
            _print_frame(analyzer.aggregate_functions(group or None, value or None))
        elif choice == "7":
            path = _read("CSV destination: ", input_func)
            if path:
                print(f"Exported to {analyzer.export_result(path)}")
        elif choice is not None:
            print("Invalid DataFrame operation choice.")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Operation failed: {exc}")


def handle_missing_data(analyzer: SalesDataAnalyzer, input_func: Callable[[str], str] = input) -> None:
    if analyzer.data.empty and len(analyzer.data.columns) == 0:
        print("Load a dataset before handling missing data.")
        return
    print("\n== Handle Missing Data ==")
    print("1. Display rows with missing values")
    print("2. Fill numeric missing values with the mean")
    print("3. Drop rows with missing values")
    print("4. Replace missing values with a specific value")
    print("5. Fill numeric missing values with the median")
    print("6. Fill missing values with the mode")
    choice = _read("Enter your choice: ", input_func)
    try:
        if choice == "1":
            result = analyzer.data[analyzer.data.isna().any(axis=1)]
            print(result.to_string(index=False) if not result.empty else "No missing values found in the dataset!")
        elif choice in {"2", "3", "5", "6"}:
            strategy = {"2": "mean", "3": "drop", "5": "median", "6": "mode"}[choice]
            summary = analyzer.clean_data(strategy)
            print(f"Cleaning complete. Missing values: {summary['missing_before']} -> {summary['missing_after']}.")
        elif choice == "4":
            value = _read("Replacement value: ", input_func)
            if value is not None:
                summary = analyzer.clean_data("value", value)
                print(f"Cleaning complete. Missing values: {summary['missing_before']} -> {summary['missing_after']}.")
        elif choice is not None:
            print("Invalid missing-data choice.")
    except (TypeError, ValueError) as exc:
        print(f"Cleaning failed: {exc}")


def descriptive_statistics(analyzer: SalesDataAnalyzer, input_func: Callable[[str], str] = input) -> None:
    try:
        result = analyzer.statistical_analysis()
    except ValueError as exc:
        print(exc)
        return
    print("\n== Descriptive Statistics ==")
    print("1. describe()")
    print("2. Means and medians")
    print("3. Standard deviations and variances")
    print("4. Quantiles")
    print("5. Missing values and correlation")
    choice = _read("Enter your choice: ", input_func)
    if choice == "1":
        _print_frame(result["describe"])
    elif choice == "2":
        print("Mean:")
        _print_frame(result["mean"])
        print("Median:")
        _print_frame(result["median"])
    elif choice == "3":
        print("Sample standard deviation:")
        _print_frame(result["std_sample"])
        print("Sample variance:")
        _print_frame(result["variance_sample"])
    elif choice == "4":
        _print_frame(result["quantiles"])
    elif choice == "5":
        print("Missing values:")
        _print_frame(result["missing"])
        print("Correlation:")
        _print_frame(result["correlation"])
    elif choice is not None:
        print("Invalid statistics choice.")


def visualize(analyzer: SalesDataAnalyzer, input_func: Callable[[str], str] = input) -> None:
    if analyzer.data.empty and len(analyzer.data.columns) == 0:
        print("Load a dataset before visualizing it.")
        return
    plot_types = ["bar", "line", "scatter", "pie", "histogram", "stack", "heatmap", "box", "subplots"]
    print("\n== Data Visualization ==")
    for index, plot_type in enumerate(plot_types, 1):
        print(f"{index}. {plot_type.title()} Plot")
    choice = _read("Enter your choice: ", input_func)
    if choice is None:
        return
    try:
        plot_type = plot_types[int(choice) - 1] if choice else None
    except (ValueError, IndexError):
        print("Invalid visualization choice.")
        return
    if plot_type in {"scatter", "bar", "line", "pie", "histogram", "stack", "box"}:
        x = _choose_column_with_input(analyzer, "X-axis/category column (blank for automatic): ", input_func, allow_blank=True)
        y = _choose_column_with_input(analyzer, "Y-axis/value column (blank for automatic): ", input_func, allow_blank=True)
        if x is None or y is None:
            return
    else:
        x = y = None
    hue = _choose_column_with_input(analyzer, "Grouping column (blank for none): ", input_func, allow_blank=True) if plot_type in {"bar", "line", "scatter", "histogram", "stack", "box"} else None
    if plot_type in {"bar", "line", "scatter", "histogram", "stack", "box"} and hue is None:
        return
    try:
        analyzer.visualize_data(plot_type, x or None, y or None, hue or None)
        print(f"{plot_type.title()} plot generated successfully.")
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Visualization failed: {exc}")


def save_visualization(analyzer: SalesDataAnalyzer, input_func: Callable[[str], str] = input) -> None:
    path = _read("Enter a filename ending in .png, .jpg, or .jpeg: ", input_func)
    if not path:
        return
    try:
        print(f"Visualization saved as {analyzer.save_visualization(path)} successfully!")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Unable to save visualization: {exc}")


def run_cli(
    analyzer: SalesDataAnalyzer | None = None,
    input_func: Callable[[str], str] = input,
) -> None:
    """Run the menu loop. ``input_func`` is injectable for CLI tests."""

    analyzer = analyzer or SalesDataAnalyzer()
    print("=========== Data Analysis & Visualization Program ===========")
    while True:
        print("\nPlease select an option:")
        print("1. Load Dataset")
        print("2. Explore Data")
        print("3. Perform DataFrame Operations")
        print("4. Handle Missing Data")
        print("5. Generate Descriptive Statistics")
        print("6. Data Visualization")
        print("7. Save Visualization")
        print("8. Exit")
        print("============================================================")
        choice = _read("Enter your choice: ", input_func)
        if choice is None:
            print("Exiting the program. Goodbye!")
            return
        if choice == "1":
            load_dataset(analyzer, input_func)
        elif choice == "2":
            explore_data(analyzer, input_func)
        elif choice == "3":
            print("DataFrame operations are available through the Python API and the detailed helper menu.")
            dataframe_operations(analyzer, input_func)
        elif choice == "4":
            handle_missing_data(analyzer, input_func)
        elif choice == "5":
            descriptive_statistics(analyzer, input_func)
        elif choice == "6":
            visualize(analyzer, input_func)
        elif choice == "7":
            save_visualization(analyzer, input_func)
        elif choice == "8":
            print("Exiting the program. Goodbye!")
            return
        else:
            print("Invalid choice. Please enter a number from 1 to 8.")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
