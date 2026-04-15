import threading
import traceback
import time
import re
import io
import base64
import pandas as pd
import numpy as np
import math
import datetime
import statistics

from models.schemas import ExecutionResult

ALLOWED_BUILTINS = {
    # ---------------------------------------------------------------------------
    # __import__ is required so that `import pandas as pd` etc. work inside exec.
    # We don't remove it — we control dangerous imports via FORBIDDEN_PATTERNS
    # (os, sys, subprocess, open, requests, socket, exec, eval) which are checked
    # BEFORE execution. This is the correct sandbox model: pattern-block first,
    # then execute with full builtins. Removing __import__ breaks ALL imports,
    # including safe ones like `import math`.
    # ---------------------------------------------------------------------------
    "__import__": __import__,
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "isinstance": isinstance,
    "type": type,
    "hasattr": hasattr,
    "getattr": getattr,
    "ValueError": ValueError,
    "KeyError": KeyError,
    "TypeError": TypeError,
    "None": None,
    "True": True,
    "False": False,
}

FORBIDDEN_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bsubprocess\b",
    r"\bopen\s*\(",
    r"\brequests\b",
    r"\bsocket\b",
    r"\b__import__\b",
    r"\.write\s*\(",
    r"\.to_csv\s*\(",
    r"\.to_excel\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
]


def _build_matplotlib_globals() -> dict:
    """
    Safely import matplotlib and seaborn using the non-interactive Agg backend.
    Returns a dict of globals to inject; returns empty dict if unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # must be set before importing pyplot
        import matplotlib.pyplot as plt
        import seaborn as sns
        return {"plt": plt, "sns": sns, "matplotlib": matplotlib}
    except ImportError:
        return {}


class _PltProxy:
    """
    Transparent proxy around matplotlib.pyplot that silences plt.show().

    The Agg backend has no GUI and emits a UserWarning on every plt.show()
    call. Since the engine captures figures via plt.get_fignums() after
    execution, show() is a no-op in this context. All other attributes
    delegate to the real pyplot module unchanged.
    """

    def __init__(self, real_plt):
        self._plt = real_plt

    def show(self, *args, **kwargs):
        # Intentional no-op: figure capture happens via get_fignums()
        pass

    def __getattr__(self, name: str):
        return getattr(self._plt, name)


class ExecutionEngine:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds
        self._mpl_globals = _build_matplotlib_globals()

    def build_safe_globals(self, df: pd.DataFrame) -> dict:
        g = {
            "__builtins__": ALLOWED_BUILTINS,
            "pd": pd,
            "np": np,
            "math": math,
            "datetime": datetime,
            "statistics": statistics,
            "re": re,
            "io": io,
            "base64": base64,
            "df": df.copy(),
        }
        g.update(self._mpl_globals)

        # Override plt.show with a silent no-op.
        # The Agg backend emits a UserWarning on every plt.show() call because
        # it has no GUI. Figures are captured via plt.get_fignums() after
        # execution, so show() is purely decorative in this context.
        if "plt" in g:
            g["plt"] = _PltProxy(g["plt"])

        return g

    def execute(self, code: str, df: pd.DataFrame) -> ExecutionResult:
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return ExecutionResult(
                    success=False,
                    output=None,
                    output_type="error",
                    error="Forbidden operation detected in code",
                    execution_time_ms=0.0,
                    serialized_output="",
                    plot_base64=None,
                )

        scope = self.build_safe_globals(df)
        start = time.time()
        success, result_value, error_msg = self._run_with_timeout(code, scope)
        elapsed_ms = (time.time() - start) * 1000

        if not success:
            return ExecutionResult(
                success=False,
                output=None,
                output_type="error",
                error=error_msg,
                execution_time_ms=elapsed_ms,
                serialized_output="",
                plot_base64=None,
            )

        # Check if matplotlib produced a figure
        plot_b64 = self._capture_plot(scope)

        result_val = scope.get("result", result_value)

        # If code generated a plot but no explicit `result`, treat plot as result
        if plot_b64 is not None and result_val is None:
            return ExecutionResult(
                success=True,
                output=None,
                output_type="plot",
                error=None,
                execution_time_ms=elapsed_ms,
                serialized_output="[plot generated]",
                plot_base64=plot_b64,
            )

        # No plot was captured AND result is None — genuine failure.
        # This is distinct from a plot query: those always produce a figure
        # via plt.figure() / sns.*. If neither happened, the code is broken.
        if result_val is None:
            return ExecutionResult(
                success=False,
                output=None,
                output_type="error",
                error="Code executed but produced no result or visualization.",
                execution_time_ms=elapsed_ms,
                serialized_output="",
                plot_base64=None,
            )

        serialized, output_type = self.serialize_result(result_val)

        return ExecutionResult(
            success=True,
            output=result_val,
            output_type=output_type,
            error=None,
            execution_time_ms=elapsed_ms,
            serialized_output=serialized,
            plot_base64=plot_b64,
        )

    # ------------------------------------------------------------------
    # Plot capture
    # ------------------------------------------------------------------

    def _capture_plot(self, scope: dict) -> str | None:
        """
        If matplotlib produced any figures during execution, save the current
        figure to a PNG buffer and return as base64.

        Uses plt.get_fignums() instead of gcf().get_axes(): more reliable because
        seaborn sometimes creates figures without adding axes to gcf() directly.
        Closes all figures after capture to free memory.
        """
        plt = scope.get("plt")
        if plt is None:
            return None
        try:
            if not plt.get_fignums():
                # No figures were created — not a plot query
                return None
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("utf-8")
            plt.close("all")
            return encoded
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
            return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_result(self, result) -> tuple[str, str]:
        import json

        if result is None:
            # Should never be reached after the explicit None-guard in execute(),
            # but kept as a hard sentinel to prevent silent success on empty results.
            return "No result returned", "error"

        if isinstance(result, pd.DataFrame):
            if len(result) == 0:
                return (
                    json.dumps({
                        "columns": list(result.columns),
                        "rows": [],
                        "shape": [0, len(result.columns)],
                    }),
                    "dataframe",
                )
            max_rows = 50
            display = result.head(max_rows)
            columns = list(display.columns)
            rows = []
            for _, row in display.iterrows():
                row_data = []
                for v in row:
                    if isinstance(v, np.integer):
                        row_data.append(int(v))
                    elif isinstance(v, np.floating):
                        row_data.append(None if np.isnan(v) else float(v))
                    elif (
                        not isinstance(v, (list, dict, str))
                        and pd.isna(v)
                    ):
                        row_data.append(None)
                    else:
                        row_data.append(
                            str(v)
                            if not isinstance(v, (int, float, bool, type(None)))
                            else v
                        )
                rows.append(row_data)
            serialized = json.dumps({
                "columns": columns,
                "rows": rows,
                "shape": [len(result), len(result.columns)],
                "truncated": len(result) > max_rows,
            })
            return serialized, "dataframe"

        if isinstance(result, pd.Series):
            lst = result.tolist()
            cleaned = []
            for v in lst:
                if isinstance(v, np.integer):
                    cleaned.append(int(v))
                elif isinstance(v, np.floating):
                    cleaned.append(None if np.isnan(v) else float(v))
                else:
                    cleaned.append(v)
            return json.dumps(cleaned), "list"

        if isinstance(result, list):
            return json.dumps(result, default=str), "list"

        if isinstance(result, np.integer):
            return str(int(result)), "scalar"

        if isinstance(result, np.floating):
            return str(float(result)), "scalar"

        return str(result), "scalar"

    # ------------------------------------------------------------------
    # Timeout runner
    # ------------------------------------------------------------------

    def _run_with_timeout(
        self, code: str, scope: dict
    ) -> tuple[bool, object, str]:
        result_container: dict = {"success": False, "error": ""}

        def target():
            try:
                exec(code, scope)  # noqa: S102
                result_container["success"] = True
            except Exception as e:
                result_container["success"] = False
                result_container["error"] = str(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            return False, None, "Execution timeout exceeded"

        if not result_container["success"]:
            return False, None, result_container["error"]

        return True, scope.get("result"), ""
