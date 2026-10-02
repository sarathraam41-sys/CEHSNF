"""
CEHSNF - Symbolic Complexity Analysis

Complexity-Aware Explainable Hybrid Symbolic-Neural Framework

This module evaluates the structural complexity of symbolic
expressions produced by symbolic regression.

The primary complexity measure is the expression complexity
reported by PySR.

Additional structural information is extracted when possible:

    - Expression string
    - Complexity
    - Number of operators
    - Number of variables
    - Number of constants
    - Expression length

This allows CEHSNF experiments to compare predictive performance
against symbolic expression complexity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


# ---------------------------------------------------------------------
# Operator definitions
# ---------------------------------------------------------------------

BINARY_OPERATORS = {
    "+",
    "-",
    "*",
    "/",
    "^",
}

UNARY_OPERATORS = {
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "abs",
    "square",
    "cube",
}

# Common mathematical constants that should not be counted as
# learned variables.
KNOWN_CONSTANTS = {
    "pi",
    "e",
}


# ---------------------------------------------------------------------
# Expression extraction
# ---------------------------------------------------------------------

def _extract_equation(
    model_or_equation: Any,
) -> str:
    """
    Extract a symbolic equation from several possible inputs.

    Supported inputs:

        1. A plain string
        2. A PySR model exposing get_best()
        3. An object exposing an equation attribute
        4. An object whose string representation is usable
    """

    # -------------------------------------------------------------
    # Plain string
    # -------------------------------------------------------------

    if isinstance(
        model_or_equation,
        str,
    ):
        equation = model_or_equation.strip()

        if not equation:
            raise ValueError(
                "The equation string cannot be empty."
            )

        return equation

    # -------------------------------------------------------------
    # PySR model
    # -------------------------------------------------------------

    if hasattr(
        model_or_equation,
        "get_best",
    ):
        try:
            best = model_or_equation.get_best()

            if isinstance(best, dict):
                equation = best.get(
                    "equation"
                )

            else:
                equation = getattr(
                    best,
                    "equation",
                    None,
                )

            if equation is not None:
                equation = str(
                    equation
                ).strip()

                if equation:
                    return equation

        except Exception:
            pass

    # -------------------------------------------------------------
    # Object with equation attribute
    # -------------------------------------------------------------

    if hasattr(
        model_or_equation,
        "equation",
    ):
        equation = getattr(
            model_or_equation,
            "equation",
        )

        if equation is not None:
            equation = str(
                equation
            ).strip()

            if equation:
                return equation

    # -------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------

    equation = str(
        model_or_equation
    ).strip()

    if not equation:
        raise ValueError(
            "Unable to extract a symbolic equation."
        )

    return equation


# ---------------------------------------------------------------------
# Operator counting
# ---------------------------------------------------------------------

def count_operators(
    equation: str,
) -> int:
    """
    Count mathematical operators in an expression.

    Both binary and common unary operators are counted.
    """

    equation = str(equation)

    count = 0

    # -------------------------------------------------------------
    # Binary operators
    # -------------------------------------------------------------

    for operator in BINARY_OPERATORS:

        count += equation.count(
            operator
        )

    # -------------------------------------------------------------
    # Unary functions
    # -------------------------------------------------------------

    for operator in UNARY_OPERATORS:

        pattern = rf"\b{re.escape(operator)}\s*\("

        count += len(
            re.findall(
                pattern,
                equation,
            )
        )

    return count


# ---------------------------------------------------------------------
# Variable counting
# ---------------------------------------------------------------------

def extract_variables(
    equation: str,
) -> list[str]:
    """
    Extract variable names from a symbolic expression.

    CEHSNF/PySR expressions commonly use variables such as:

        x0
        x1
        x2
        x3

    Returns unique variables in sorted order.
    """

    equation = str(equation)

    variables = re.findall(
        r"\bx\d+\b",
        equation,
    )

    return sorted(
        set(variables),
        key=lambda value: (
            len(value),
            int(value[1:]),
        ),
    )


# ---------------------------------------------------------------------
# Constant counting
# ---------------------------------------------------------------------

def extract_constants(
    equation: str,
) -> list[str]:
    """
    Extract numerical constants from an expression.

    Integer and floating-point constants are supported.

    Variable indices such as x0/x1 are excluded.
    """

    equation = str(equation)

    constants = re.findall(
        r"(?<![A-Za-z0-9_])"
        r"[-+]?"
        r"(?:"
        r"\d+\.\d*"
        r"|"
        r"\.\d+"
        r"|"
        r"\d+"
        r")"
        r"(?:[eE][-+]?\d+)?",
        equation,
    )

    return constants


# ---------------------------------------------------------------------
# Expression length
# ---------------------------------------------------------------------

def expression_length(
    equation: str,
) -> int:
    """
    Return the number of characters in the symbolic expression.
    """

    return len(
        str(equation).strip()
    )


# ---------------------------------------------------------------------
# PySR complexity
# ---------------------------------------------------------------------

def get_pysr_complexity(
    model: Any,
) -> Optional[float]:
    """
    Extract the complexity reported by PySR.

    Returns None if the supplied object does not expose a
    PySR-style complexity value.
    """

    if not hasattr(
        model,
        "get_best",
    ):
        return None

    try:
        best = model.get_best()

        if isinstance(best, dict):
            complexity = best.get(
                "complexity"
            )

        else:
            complexity = getattr(
                best,
                "complexity",
                None,
            )

        if complexity is None:
            return None

        return float(
            complexity
        )

    except Exception:
        return None


# ---------------------------------------------------------------------
# Complete complexity analysis
# ---------------------------------------------------------------------

def analyze_complexity(
    model_or_equation: Any,
    complexity: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a complete symbolic complexity report.

    Parameters
    ----------
    model_or_equation:
        PySR model or symbolic equation string.

    complexity:
        Optional manually supplied complexity value.

        If omitted, the function attempts to extract the
        complexity directly from a PySR model.

    Returns
    -------
    dict
        Complete symbolic complexity information.
    """

    equation = _extract_equation(
        model_or_equation
    )

    if complexity is None:
        complexity = get_pysr_complexity(
            model_or_equation
        )

    variables = extract_variables(
        equation
    )

    constants = extract_constants(
        equation
    )

    operators = count_operators(
        equation
    )

    length = expression_length(
        equation
    )

    return {
        "equation": equation,
        "complexity": complexity,
        "operator_count": operators,
        "variable_count": len(variables),
        "variables": variables,
        "constant_count": len(constants),
        "constants": constants,
        "expression_length": length,
    }


# ---------------------------------------------------------------------
# Compare symbolic expressions
# ---------------------------------------------------------------------

def compare_complexity(
    expressions: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple symbolic expressions.

    Parameters
    ----------
    expressions:
        Dictionary mapping model names to:

            - equation strings
            - PySR models

    Returns
    -------
    dict
        Complexity report for each model.
    """

    if not expressions:
        raise ValueError(
            "expressions must contain at least one entry."
        )

    results: Dict[str, Dict[str, Any]] = {}

    for model_name, expression in expressions.items():

        results[str(model_name)] = (
            analyze_complexity(
                expression
            )
        )

    return results


# ---------------------------------------------------------------------
# Complexity table
# ---------------------------------------------------------------------

def complexity_table(
    expressions: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """
    Convert complexity analysis into table-style rows.
    """

    comparison = compare_complexity(
        expressions
    )

    rows: list[Dict[str, Any]] = []

    for model_name, report in comparison.items():

        rows.append(
            {
                "model": model_name,
                "equation": report["equation"],
                "complexity": report["complexity"],
                "operator_count": report[
                    "operator_count"
                ],
                "variable_count": report[
                    "variable_count"
                ],
                "constant_count": report[
                    "constant_count"
                ],
                "expression_length": report[
                    "expression_length"
                ],
            }
        )

    return rows


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

__all__ = [
    "count_operators",
    "extract_variables",
    "extract_constants",
    "expression_length",
    "get_pysr_complexity",
    "analyze_complexity",
    "compare_complexity",
    "complexity_table",
]


# ---------------------------------------------------------------------
# Module test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "CEHSNF symbolic complexity test"
    )

    print("-" * 50)

    # -------------------------------------------------------------
    # Example symbolic expressions
    # -------------------------------------------------------------

    expressions = {
        "simple": "x0 * -1.0",

        "moderate": (
            "x0 * (x1 + -0.5)"
        ),

        "complex": (
            "sin(x0) * "
            "(x1 + 0.5) / "
            "(cos(x2) - 0.25)"
        ),
    }

    # -------------------------------------------------------------
    # Individual analysis
    # -------------------------------------------------------------

    for name, equation in expressions.items():

        report = analyze_complexity(
            equation
        )

        print()
        print(name)
        print("-" * 50)

        print(
            f"Equation          : "
            f"{report['equation']}"
        )

        print(
            f"Complexity        : "
            f"{report['complexity']}"
        )

        print(
            f"Operators         : "
            f"{report['operator_count']}"
        )

        print(
            f"Variables         : "
            f"{report['variable_count']}"
        )

        print(
            f"Variable names    : "
            f"{report['variables']}"
        )

        print(
            f"Constants         : "
            f"{report['constant_count']}"
        )

        print(
            f"Expression length : "
            f"{report['expression_length']}"
        )

    # -------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------

    comparison = compare_complexity(
        expressions
    )

    print()
    print(
        "Complexity comparison"
    )
    print("-" * 50)

    for name, report in comparison.items():

        print(
            f"{name:10s} | "
            f"operators={report['operator_count']} | "
            f"variables={report['variable_count']} | "
            f"constants={report['constant_count']} | "
            f"length={report['expression_length']}"
        )

    # -------------------------------------------------------------
    # Table
    # -------------------------------------------------------------

    rows = complexity_table(
        expressions
    )

    print()
    print("Complexity table")
    print("-" * 50)

    for row in rows:

        print(
            f"{row['model']:10s} | "
            f"complexity={row['complexity']} | "
            f"operators={row['operator_count']} | "
            f"variables={row['variable_count']} | "
            f"constants={row['constant_count']} | "
            f"length={row['expression_length']}"
        )

    print()
    print(
        "Complexity analysis test "
        "completed successfully."
    )