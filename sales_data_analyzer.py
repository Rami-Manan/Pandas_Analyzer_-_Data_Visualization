"""Reusable sales-data analysis and visualization functionality.

The :class:`SalesDataAnalyzer` class keeps data operations separate from the
interactive command-line interface in ``main.py``.  Methods return useful
objects so the same functionality can be used from a notebook or automated
tests without scraping terminal output.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
import numpy as np
import pandas as pd
import seaborn as sns


class SalesDataAnalyzer:
    """Analyze, transform, and visualize a CSV-backed sales DataFrame."""

    SUPPORTED_PLOT_TYPES = {
        "bar",
        "line",
        "scatter",
        "pie",
        "histogram",
        "stack",
        "heatmap",
        "box",
        "subplots",
    }

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        """Initialize the analyzer with a defensive copy of ``data``."""

        self.data = data.copy(deep=True) if data is not None else pd.DataFrame()
        if len(self.data.columns):
            self.data.columns = self._make_unique_columns(self.data.columns)
        self.current_figure: plt.Figure | None = None
        self.last_result: Any = None

    def __del__(self) -> None:
        """Release a Matplotlib figure if one is still owned by the analyzer."""

        figure = getattr(self, "current_figure", None)
        if figure is not None:
            try:
                plt.close(figure)
            except (RuntimeError, TypeError):
                pass

    def _require_data(self) -> pd.DataFrame:
        if self.data.empty and len(self.data.columns) == 0:
            raise ValueError("No dataset is loaded. Load a CSV file first.")
        return self.data

    def _require_columns(self, columns: Iterable[str]) -> None:
        missing = [column for column in columns if column not in self.data.columns]
        if missing:
            raise KeyError(f"Unknown column(s): {', '.join(map(str, missing))}")

    @staticmethod
    def _is_missing_scalar(value: Any) -> bool:
        """Return whether ``value`` is a scalar missing value."""

        if value is None:
            return True
        missing = pd.isna(value)
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False

    @staticmethod
    def _coerce_fill_value(series: pd.Series, value: Any, column: str) -> Any:
        """Convert a fill value to a dtype-compatible scalar."""

        if SalesDataAnalyzer._is_missing_scalar(value):
            raise ValueError(f"Replacement value for '{column}' must not be missing.")
        if pd.api.types.is_numeric_dtype(series):
            try:
                return pd.to_numeric(pd.Series([value]), errors="raise").iloc[0]
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Replacement value for numeric column '{column}' must be numeric.") from exc
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.to_datetime(value, errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Replacement value for date column '{column}' must be a valid date.") from exc
        if pd.api.types.is_bool_dtype(series) and isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n"}:
                return False
            raise ValueError(f"Replacement value for boolean column '{column}' must be true or false.")
        return value

    @classmethod
    def _fill_missing_values(cls, data: pd.DataFrame, fill_value: Any) -> pd.DataFrame:
        """Fill missing values without silently changing numeric/date dtypes."""

        values_by_column = fill_value if isinstance(fill_value, Mapping) else None
        if values_by_column is not None:
            unknown = set(values_by_column).difference(data.columns)
            if unknown:
                raise KeyError(f"Unknown replacement column(s): {', '.join(map(str, unknown))}")

        updated = data.copy(deep=True)
        for column in updated.columns:
            if not updated[column].isna().any():
                continue
            value = values_by_column[column] if values_by_column is not None and column in values_by_column else fill_value
            if values_by_column is not None and column not in values_by_column:
                continue
            converted = cls._coerce_fill_value(updated[column], value, column)
            series = updated[column]
            if isinstance(series.dtype, pd.CategoricalDtype) and converted not in series.cat.categories:
                series = series.cat.add_categories([converted])
            updated[column] = series.fillna(converted)
        return updated

    @property
    def numeric_columns(self) -> list[str]:
        """Return numeric columns in their original DataFrame order."""

        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    @property
    def categorical_columns(self) -> list[str]:
        """Return non-numeric columns in their original DataFrame order."""

        return self.data.select_dtypes(exclude=[np.number]).columns.tolist()

    def load_data(self, file_path: str | Path) -> pd.DataFrame:
        """Load a non-empty CSV file, replacing the active dataset on success."""

        path = Path(file_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Dataset path is not a file: {path}")

        try:
            loaded = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The CSV file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"The CSV file is malformed: {exc}") from exc
        except PermissionError:
            raise
        except UnicodeDecodeError as exc:
            raise ValueError("The CSV file is not valid UTF-8 text.") from exc

        if loaded.shape[1] == 0:
            raise ValueError("The CSV file has no columns.")
        if loaded.empty:
            raise ValueError("The CSV file contains headers but no data rows.")

        loaded.columns = self._make_unique_columns(loaded.columns)
        self.data = loaded
        self.last_result = loaded.copy(deep=True)
        return self.data.copy(deep=True)

    @staticmethod
    def _make_unique_columns(columns: Iterable[object]) -> list[str]:
        """Normalize column labels and suffix duplicates rather than dropping them."""

        counts: dict[str, int] = {}
        used: set[str] = set()
        result: list[str] = []
        for raw_column in columns:
            base = str(raw_column).strip() or "unnamed"
            count = counts.get(base, 0)
            candidate = base if count == 0 else f"{base}_{count}"
            while candidate in used:
                count += 1
                candidate = f"{base}_{count}"
            result.append(candidate)
            used.add(candidate)
            counts[base] = count + 1
        return result

    def explore_data(self) -> dict[str, Any]:
        """Return common structural summaries for the active dataset."""

        data = self._require_data()
        buffer = io.StringIO()
        data.info(buf=buffer)
        result = {
            "head": data.head(),
            "tail": data.tail(),
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.astype(str).to_dict(),
            "shape": data.shape,
            "info": buffer.getvalue(),
            "missing": data.isna().sum(),
        }
        self.last_result = result
        return result

    def data_quality_report(self) -> pd.DataFrame:
        """Return one row per column with completeness and type information."""

        data = self._require_data()
        row_count = len(data)
        report = pd.DataFrame(
            {
                "dtype": data.dtypes.astype(str),
                "non_null": data.notna().sum(),
                "missing": data.isna().sum(),
                "unique": data.nunique(dropna=True),
                "is_numeric": [pd.api.types.is_numeric_dtype(data[column]) for column in data.columns],
            }
        )
        report["missing_pct"] = (
            report["missing"].div(row_count).mul(100).round(2) if row_count else 0.0
        )
        report = report.reset_index(names="column")
        self.last_result = report
        return report

    def clean_data(
        self,
        strategy: str = "report",
        fill_value: Any = None,
        convert_dates: bool = True,
        remove_duplicates: bool = True,
    ) -> dict[str, Any]:
        """Clean missing values and duplicates using an explicit strategy.

        ``strategy`` is one of ``report``, ``mean``, ``median``, ``mode``,
        ``drop``, or ``value``.  The active DataFrame changes only after all
        requested operations succeed.
        """

        data = self._require_data().copy(deep=True)
        before_rows = len(data)
        data.columns = self._make_unique_columns(data.columns)
        data = data.replace([np.inf, -np.inf], np.nan)

        if convert_dates:
            for column in data.columns:
                if column.lower() == "date" or column.lower().endswith("_date"):
                    converted = pd.to_datetime(data[column], errors="coerce")
                    if converted.notna().any():
                        data[column] = converted

        duplicate_count = int(data.duplicated().sum())
        if remove_duplicates and duplicate_count:
            data = data.drop_duplicates().reset_index(drop=True)

        missing_before = int(data.isna().sum().sum())
        strategy = strategy.lower().strip()
        if strategy == "report":
            pass
        elif strategy == "drop":
            data = data.dropna().reset_index(drop=True)
        elif strategy == "mean":
            for column in data.select_dtypes(include=[np.number]).columns:
                data[column] = data[column].fillna(data[column].mean())
        elif strategy == "median":
            for column in data.select_dtypes(include=[np.number]).columns:
                data[column] = data[column].fillna(data[column].median())
        elif strategy == "mode":
            for column in data.columns:
                modes = data[column].mode(dropna=True)
                if not modes.empty:
                    data[column] = data[column].fillna(modes.iloc[0])
        elif strategy == "value":
            if fill_value is None:
                raise ValueError("fill_value is required when strategy='value'.")
            data = self._fill_missing_values(data, fill_value)
        else:
            raise ValueError(
                "Unknown cleaning strategy. Use report, mean, median, mode, drop, or value."
            )

        self.data = data
        result = {
            "data": data.copy(deep=True),
            "rows_before": before_rows,
            "rows_after": len(data),
            "duplicates_found": duplicate_count,
            "missing_before": missing_before,
            "missing_after": int(data.isna().sum().sum()),
            "strategy": strategy,
        }
        self.last_result = result
        return result

    def numpy_operations(self, column: str | None = None) -> dict[str, np.ndarray]:
        """Convert a numeric column to an array and demonstrate indexing/slicing."""

        data = self._require_data()
        if column is None:
            if not self.numeric_columns:
                raise ValueError("The dataset has no numeric columns.")
            column = self.numeric_columns[0]
        self._require_columns([column])
        values = pd.to_numeric(data[column], errors="coerce").dropna().to_numpy()
        result = {
            "array": values,
            "first": values[0:1],
            "slice": values[: min(5, len(values))],
            "doubled": values * 2,
        }
        self.last_result = result
        return result

    def mathematical_operations(
        self,
        left_column: str | None = None,
        right_column: str | None = None,
        operation: str = "add",
        result_column: str | None = None,
    ) -> pd.Series:
        """Perform safe element-wise arithmetic on two numeric columns."""

        data = self._require_data()
        numeric = self.numeric_columns
        if left_column is None or right_column is None:
            if len(numeric) < 2:
                raise ValueError("At least two numeric columns are required.")
            left_column, right_column = numeric[:2]
        self._require_columns([left_column, right_column])
        left = pd.to_numeric(data[left_column], errors="coerce")
        right = pd.to_numeric(data[right_column], errors="coerce")
        operation = operation.lower().strip()
        if operation == "add":
            result = left + right
        elif operation == "subtract":
            result = left - right
        elif operation == "multiply":
            result = left * right
        elif operation == "divide":
            result = left.div(right.replace(0, np.nan))
        else:
            raise ValueError("operation must be add, subtract, multiply, or divide.")

        result = result.replace([np.inf, -np.inf], np.nan)
        if result_column:
            updated = data.copy(deep=True)
            updated[result_column] = result
            self.data = updated
        self.last_result = result
        return result

    def combine_data(
        self,
        other_dataframe: pd.DataFrame,
        how: str = "concat",
        on: str | list[str] | None = None,
        join: str = "outer",
    ) -> pd.DataFrame:
        """Combine the active data with another DataFrame using concat/merge/join."""

        if not isinstance(other_dataframe, pd.DataFrame):
            raise TypeError("other_dataframe must be a pandas DataFrame.")
        current = self._require_data()
        how = how.lower().strip()
        if how == "concat":
            combined = pd.concat([current, other_dataframe], ignore_index=True, sort=False)
        elif how == "merge":
            if on is None:
                raise ValueError("on is required for a merge.")
            keys = [on] if isinstance(on, str) else list(on)
            if not set(keys).issubset(current.columns) or not set(keys).issubset(other_dataframe.columns):
                raise KeyError("All merge keys must exist in both DataFrames.")
            combined = current.merge(other_dataframe, how=join, on=keys)
        elif how == "join":
            combined = current.join(other_dataframe, how=join, lsuffix="_left", rsuffix="_right")
        else:
            raise ValueError("how must be concat, merge, or join.")
        self.data = combined
        self.last_result = combined
        return combined.copy(deep=True)

    def split_data(
        self,
        column: str | None = None,
        values: Iterable[Any] | None = None,
    ) -> dict[Any, pd.DataFrame]:
        """Split the dataset into DataFrames keyed by distinct column values."""

        data = self._require_data()
        if column is None:
            candidates = [c for c in self.categorical_columns if data[c].nunique(dropna=True) > 1]
            if not candidates:
                raise ValueError("No suitable categorical column is available for splitting.")
            column = candidates[0]
        self._require_columns([column])
        selected = set(values) if values is not None else set(data[column].dropna().unique())
        result = {
            value: group.copy(deep=True)
            for value, group in data.groupby(column, dropna=False, sort=False)
            if value in selected
        }
        self.last_result = result
        return result

    def search_sort_filter(
        self,
        search_column: str | None = None,
        search_value: Any = None,
        sort_by: str | None = None,
        ascending: bool = True,
        filters: Mapping[str, Any] | None = None,
        numeric_range: tuple[float | None, float | None] | None = None,
        date_range: tuple[Any, Any] | None = None,
    ) -> pd.DataFrame:
        """Search, sort, and filter records without mutating the active data."""

        result = self._require_data().copy(deep=True)
        range_requested = numeric_range is not None or date_range is not None
        if search_value is not None and search_column is None:
            raise ValueError("search_value requires search_column to identify the field.")
        if search_column is not None and search_value is None and not range_requested:
            raise ValueError("search_value is required when search_column is supplied.")
        if search_column is not None:
            self._require_columns([search_column])
            series = result[search_column]
            if search_value is not None and pd.api.types.is_numeric_dtype(series):
                numeric_value = pd.to_numeric(pd.Series([search_value]), errors="coerce").iloc[0]
                if pd.isna(numeric_value):
                    raise ValueError("Numeric searches require a numeric search_value.")
                mask = series == numeric_value
                result = result.loc[mask]
            elif search_value is not None:
                mask = series.astype("string").str.contains(
                    str(search_value), case=False, na=False, regex=False
                )
                result = result.loc[mask]

        if filters:
            self._require_columns(filters.keys())
            for column, expected in filters.items():
                if expected is None:
                    result = result[result[column].isna()]
                elif isinstance(expected, (list, tuple, set, np.ndarray, pd.Series, pd.Index)):
                    result = result[result[column].isin(expected)]
                else:
                    result = result[result[column] == expected]

        if numeric_range is not None:
            if search_column is None:
                raise ValueError("numeric_range requires search_column to identify the numeric field.")
            try:
                lower, upper = numeric_range
            except (TypeError, ValueError) as exc:
                raise ValueError("numeric_range must contain exactly two bounds.") from exc
            if lower is not None:
                lower = pd.to_numeric(pd.Series([lower]), errors="coerce").iloc[0]
            if upper is not None:
                upper = pd.to_numeric(pd.Series([upper]), errors="coerce").iloc[0]
            if (lower is not None and pd.isna(lower)) or (upper is not None and pd.isna(upper)):
                raise TypeError("numeric_range bounds must be numeric or None.")
            if lower is not None and upper is not None and lower > upper:
                raise ValueError("numeric_range lower bound cannot exceed the upper bound.")
            numeric_values = pd.to_numeric(result[search_column], errors="coerce")
            if numeric_values.notna().sum() == 0:
                raise TypeError("numeric_range can only be used with a numeric column.")
            numeric_mask = numeric_values.notna()
            if lower is not None:
                numeric_mask &= numeric_values >= lower
            if upper is not None:
                numeric_mask &= numeric_values <= upper
            result = result.loc[numeric_mask]

        if date_range is not None:
            if search_column is None:
                raise ValueError("date_range requires search_column to identify the date field.")
            try:
                start, end = date_range
            except (TypeError, ValueError) as exc:
                raise ValueError("date_range must contain exactly two date bounds.") from exc
            try:
                start = pd.to_datetime(start, errors="raise")
                end = pd.to_datetime(end, errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError("date_range bounds must be valid dates.") from exc
            if start > end:
                raise ValueError("date_range start cannot be after the end.")
            dates = pd.to_datetime(result[search_column], errors="coerce")
            if dates.notna().sum() == 0:
                raise TypeError("date_range can only be used with a date-like column.")
            result = result.loc[dates.between(start, end, inclusive="both")]

        if sort_by is not None:
            self._require_columns([sort_by])
            result = result.sort_values(sort_by, ascending=ascending, kind="stable")
        result = result.reset_index(drop=True)
        self.last_result = result
        return result

    def aggregate_functions(
        self,
        group_by: str | list[str] | None = None,
        value_column: str | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Calculate common aggregations globally or for each group."""

        data = self._require_data()
        if value_column is not None:
            self._require_columns([value_column])
            if not pd.api.types.is_numeric_dtype(data[value_column]):
                raise TypeError("value_column must contain numeric values for sum/mean/min/max aggregation.")
        if group_by is None:
            numeric = [value_column] if value_column else self.numeric_columns
            if not numeric:
                raise ValueError("No numeric column is available for aggregation.")
            if value_column is not None and not pd.api.types.is_numeric_dtype(data[value_column]):
                raise TypeError("value_column must contain numeric values for aggregation.")
            result = data[numeric].agg(["sum", "mean", "count", "min", "max"])
        else:
            group_columns = [group_by] if isinstance(group_by, str) else list(group_by)
            if not group_columns:
                raise ValueError("group_by must contain at least one column.")
            self._require_columns(group_columns)
            target = value_column or (self.numeric_columns[0] if self.numeric_columns else None)
            if target is None:
                result = data.groupby(group_columns, dropna=False).size().to_frame("count")
            else:
                result = data.groupby(group_columns, dropna=False)[target].agg(
                    ["sum", "mean", "count", "min", "max"]
                )
        self.last_result = result
        return result

    def statistical_analysis(self) -> dict[str, Any]:
        """Return descriptive, distribution, missingness, and correlation statistics."""

        data = self._require_data()
        numeric = data.select_dtypes(include=[np.number])
        result = {
            "describe": data.describe(include="all"),
            "mean": numeric.mean(),
            "median": numeric.median(),
            "mode": numeric.mode().iloc[0] if not numeric.mode().empty else pd.Series(dtype=float),
            "std_sample": numeric.std(ddof=1),
            "variance_sample": numeric.var(ddof=1),
            "min": numeric.min(),
            "max": numeric.max(),
            "quantiles": numeric.quantile([0.25, 0.5, 0.75]),
            "count": data.count(),
            "missing": data.isna().sum(),
            "correlation": numeric.corr() if len(numeric.columns) >= 2 else pd.DataFrame(),
        }
        self.last_result = result
        return result

    def create_pivot_table(
        self,
        index: str | list[str],
        columns: str | list[str] | None = None,
        values: str | list[str] | None = None,
        aggfunc: str | list[str] | Mapping[str, str] = "mean",
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """Create a validated Pandas pivot table."""

        data = self._require_data()
        index_columns = [index] if isinstance(index, str) else list(index)
        all_columns = index_columns
        if columns is not None:
            all_columns += [columns] if isinstance(columns, str) else list(columns)
        if values is not None:
            all_columns += [values] if isinstance(values, str) else list(values)
        self._require_columns(all_columns)
        result = pd.pivot_table(
            data,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value,
        )
        self.last_result = result
        return result

    def groupby_transform(
        self,
        group_by: str,
        value_column: str,
        method: str = "mean",
        result_column: str | None = None,
    ) -> pd.Series:
        """Apply a group-wise transform such as mean, sum, or rank."""

        self._require_data()
        self._require_columns([group_by, value_column])
        method = method.lower().strip()
        if method not in {"mean", "sum", "count", "min", "max", "rank"}:
            raise ValueError("method must be mean, sum, count, min, max, or rank.")
        if method in {"mean", "sum", "rank"} and not pd.api.types.is_numeric_dtype(self.data[value_column]):
            raise TypeError(f"value_column must be numeric for the '{method}' transform.")
        grouped = self.data.groupby(group_by)[value_column]
        result = grouped.transform(method)
        if result_column:
            updated = self.data.copy(deep=True)
            updated[result_column] = result
            self.data = updated
        self.last_result = result
        return result

    def reindex_and_rename(
        self,
        labels: Iterable[Any] | None = None,
        rename: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """Return and activate a copy with optional row reindexing and renaming."""

        updated = self._require_data().copy(deep=True)
        if rename:
            updated = updated.rename(columns=dict(rename))
        if labels is not None:
            updated = updated.reindex(labels)
        self.data = updated
        self.last_result = updated
        return updated.copy(deep=True)

    def visualize_data(
        self,
        plot_type: str,
        x: str | None = None,
        y: str | None = None,
        hue: str | None = None,
        title: str | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> plt.Figure:
        """Create one of the supported Matplotlib or Seaborn visualizations."""

        data = self._require_data()
        plot_type = plot_type.lower().strip().replace("_", " ")
        plot_type = {
            "bar chart": "bar",
            "line chart": "line",
            "scatter plot": "scatter",
            "pie chart": "pie",
            "histogram plot": "histogram",
            "stacked plot": "stack",
            "stacked chart": "stack",
            "heat map": "heatmap",
            "box plot": "box",
            "subplot": "subplots",
        }.get(plot_type, plot_type)
        if plot_type not in self.SUPPORTED_PLOT_TYPES:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        if data.empty:
            raise ValueError("Cannot visualize an empty dataset.")
        if x:
            self._require_columns([x])
        if y:
            self._require_columns([y])
        if hue:
            self._require_columns([hue])

        if self.current_figure is not None:
            plt.close(self.current_figure)
            self.current_figure = None

        def frame_for(*columns: str | None) -> pd.DataFrame:
            selected = list(dict.fromkeys(column for column in columns if column is not None))
            frame = data.loc[:, selected].dropna()
            if frame.empty:
                raise ValueError("The selected visualization columns contain no complete rows.")
            return frame

        def add_row_number(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
            row_column = "__row_number__"
            while row_column in frame.columns:
                row_column = f"_{row_column}"
            updated = frame.copy()
            updated[row_column] = np.arange(len(updated))
            return updated, row_column

        figure: plt.Figure | None = None
        try:
            figure, axis = plt.subplots(figsize=(9, 5))
            sns.set_theme(style="whitegrid")
            if plot_type == "bar":
                y = y or (self.numeric_columns[0] if self.numeric_columns else None)
                if y is None:
                    raise ValueError("Bar plots require a numeric y column.")
                if not pd.api.types.is_numeric_dtype(data[y]):
                    raise TypeError("Bar plots require a numeric y column.")
                x = x or (self.categorical_columns[0] if self.categorical_columns else None)
                if x is None:
                    frame = frame_for(y, hue)
                    frame, row_column = add_row_number(frame)
                    sns.barplot(data=frame, x=row_column, y=y, hue=hue, ax=axis, errorbar=None)
                    axis.set_xlabel("Row")
                else:
                    frame = frame_for(x, y, hue)
                    sns.barplot(data=frame, x=x, y=y, hue=hue, ax=axis, errorbar=None)
            elif plot_type == "line":
                y = y or (self.numeric_columns[0] if self.numeric_columns else None)
                if y is None:
                    raise ValueError("Line plots require a numeric y column.")
                if not pd.api.types.is_numeric_dtype(data[y]):
                    raise TypeError("Line plots require a numeric y column.")
                x = x or (self.categorical_columns[0] if self.categorical_columns else None)
                if x is None:
                    frame = frame_for(y, hue)
                    frame, row_column = add_row_number(frame)
                    sns.lineplot(data=frame, x=row_column, y=y, hue=hue, marker="o", ax=axis)
                    axis.set_xlabel("Row")
                else:
                    frame = frame_for(x, y, hue)
                    sns.lineplot(data=frame, x=x, y=y, hue=hue, marker="o", ax=axis)
            elif plot_type == "scatter":
                if not x or not y:
                    raise ValueError("Scatter plots require x and y columns.")
                if not pd.api.types.is_numeric_dtype(data[x]) or not pd.api.types.is_numeric_dtype(data[y]):
                    raise TypeError("Scatter plot axes must be numeric.")
                frame = frame_for(x, y, hue)
                sns.scatterplot(data=frame, x=x, y=y, hue=hue, ax=axis)
            elif plot_type == "pie":
                x = x or (self.categorical_columns[0] if self.categorical_columns else None)
                y = y or (self.numeric_columns[0] if self.numeric_columns else None)
                if x is None or y is None:
                    raise ValueError("Pie charts require categorical x and numeric y columns.")
                if not pd.api.types.is_numeric_dtype(data[y]):
                    raise TypeError("Pie charts require a numeric y column.")
                frame = frame_for(x, y)
                grouped = frame.groupby(x, sort=False)[y].sum().sort_values(ascending=False)
                if grouped.empty or (grouped < 0).any() or grouped.sum() <= 0:
                    raise ValueError("Pie charts require positive, non-negative aggregated values.")
                axis.pie(grouped.values, labels=grouped.index.astype(str), autopct="%1.1f%%")
                axis.set_ylabel("")
            elif plot_type == "histogram":
                candidate = None
                if x and pd.api.types.is_numeric_dtype(data[x]):
                    candidate = x
                elif y and pd.api.types.is_numeric_dtype(data[y]):
                    candidate = y
                elif not x and not y and self.numeric_columns:
                    candidate = self.numeric_columns[0]
                if candidate is None:
                    raise TypeError("Histograms require a numeric x or y column.")
                frame = frame_for(candidate, hue)
                sns.histplot(data=frame, x=candidate, hue=hue, kde=False, ax=axis)
            elif plot_type == "stack":
                y = y or (self.numeric_columns[0] if self.numeric_columns else None)
                if y is None:
                    raise ValueError("Stacked charts require a numeric y column.")
                if not pd.api.types.is_numeric_dtype(data[y]):
                    raise TypeError("Stacked charts require a numeric y column.")
                x = x or (self.categorical_columns[0] if self.categorical_columns else None)
                if x is None:
                    frame = frame_for(y, hue)
                    frame, row_column = add_row_number(frame)
                    if hue:
                        table = frame.pivot_table(index=row_column, columns=hue, values=y, aggfunc="sum", fill_value=0)
                    else:
                        table = frame.set_index(row_column)[[y]]
                else:
                    frame = frame_for(x, y, hue)
                    table = frame.pivot_table(index=x, columns=hue, values=y, aggfunc="sum", fill_value=0) if hue else frame.groupby(x, sort=False)[y].sum().to_frame()
                table.plot(kind="bar", stacked=True, ax=axis)
            elif plot_type == "heatmap":
                numeric = data.select_dtypes(include=[np.number])
                if numeric.shape[1] < 2:
                    raise ValueError("Heatmaps require at least two numeric columns.")
                # Seaborn 0.13 currently calls Matplotlib's deprecated ``set_bad``
                # internally. Keep the compatibility warning local to this call so
                # applications using warnings-as-errors do not fail on a valid plot.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
                    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, ax=axis)
            elif plot_type == "box":
                y = y or (self.numeric_columns[0] if self.numeric_columns else None)
                if y is None:
                    raise ValueError("Box plots require a numeric y column.")
                if not pd.api.types.is_numeric_dtype(data[y]):
                    raise TypeError("Box plots require a numeric y column.")
                frame = frame_for(x, y, hue)
                # Matplotlib 3.11 warns about the ``vert`` keyword used by the
                # currently supported Seaborn release. Suppress only that known
                # third-party compatibility warning.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
                    sns.boxplot(data=frame, x=x, y=y, hue=hue, ax=axis)
            elif plot_type == "subplots":
                plt.close(figure)
                figure, axes = plt.subplots(1, 2, figsize=(12, 5))
                numeric = self.numeric_columns
                if len(numeric) < 2:
                    raise ValueError("Subplots require at least two numeric columns.")
                frame = data[numeric[:2]].dropna()
                if frame.empty:
                    raise ValueError("The selected visualization columns contain no complete rows.")
                sns.histplot(data=frame, x=numeric[0], ax=axes[0], color="steelblue")
                sns.scatterplot(data=frame, x=numeric[0], y=numeric[1], ax=axes[1])
                axes[0].set_title(f"Distribution of {numeric[0]}")
                axes[1].set_title(f"{numeric[0]} vs {numeric[1]}")

            figure.suptitle(title or f"{plot_type.title()} Plot")
            figure.tight_layout(rect=(0, 0, 1, 0.96))
        except Exception:
            if figure is not None:
                plt.close(figure)
            self.current_figure = None
            raise

        self.current_figure = figure
        if save_path is not None:
            self.save_visualization(save_path)
        if show:
            plt.show()
        return figure

    def save_visualization(self, file_path: str | Path) -> Path:
        """Save the current figure to PNG, JPEG, or JPG after validation."""

        if self.current_figure is None:
            raise RuntimeError("No visualization has been generated yet.")
        path = Path(file_path).expanduser()
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("Visualization format must be .png, .jpg, or .jpeg.")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.current_figure.savefig(path, bbox_inches="tight")
        except (OSError, ValueError) as exc:
            raise OSError(f"Could not save visualization to {path}: {exc}") from exc
        return path

    def export_result(self, file_path: str | Path, result: Any = None) -> Path:
        """Export a DataFrame or Series result to CSV."""

        value = self.last_result if result is None else result
        if isinstance(value, pd.Series):
            value = value.to_frame()
        if not isinstance(value, pd.DataFrame):
            raise TypeError("The result to export must be a pandas DataFrame or Series.")
        path = Path(file_path).expanduser()
        if path.suffix.lower() != ".csv":
            raise ValueError("Analysis result exports must use the .csv extension.")
        path.parent.mkdir(parents=True, exist_ok=True)
        value.to_csv(path, index=True)
        return path
