# solvers.py
# The "math engine" of the agent. These are plain functions (no input() calls)
# so the AI agent can call them with data it extracted from the user's message.

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment


# ---------------------------------------------------------------
# TRANSPORTATION METHODS  (each returns an allocation matrix)
# ---------------------------------------------------------------
def northwest_corner(cost, supply, demand):
    m, n = len(supply), len(demand)
    s, d = supply[:], demand[:]
    alloc = [[0] * n for _ in range(m)]
    i, j = 0, 0
    while i < m and j < n:
        q = min(s[i], d[j])
        alloc[i][j] = q
        s[i] -= q
        d[j] -= q
        if s[i] == 0:
            i += 1
        else:
            j += 1
    return alloc


def least_cost(cost, supply, demand):
    m, n = len(supply), len(demand)
    s, d = supply[:], demand[:]
    alloc = [[0] * n for _ in range(m)]
    while sum(s) > 0:
        best_i, best_j, best_cost = None, None, float("inf")
        for i in range(m):
            for j in range(n):
                if s[i] > 0 and d[j] > 0 and cost[i][j] < best_cost:
                    best_i, best_j, best_cost = i, j, cost[i][j]
        q = min(s[best_i], d[best_j])
        alloc[best_i][best_j] = q
        s[best_i] -= q
        d[best_j] -= q
    return alloc


def penalty(values):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    return values[1] - values[0]


def vogel(cost, supply, demand):
    m, n = len(supply), len(demand)
    s, d = supply[:], demand[:]
    alloc = [[0] * n for _ in range(m)]
    rows, cols = list(range(m)), list(range(n))
    while rows and cols:
        row_pen = [penalty([cost[i][j] for j in cols]) for i in rows]
        col_pen = [penalty([cost[i][j] for i in rows]) for j in cols]
        if max(row_pen) >= max(col_pen):
            i = rows[row_pen.index(max(row_pen))]
            j = min(cols, key=lambda c: cost[i][c])
        else:
            j = cols[col_pen.index(max(col_pen))]
            i = min(rows, key=lambda r: cost[r][j])
        q = min(s[i], d[j])
        alloc[i][j] += q
        s[i] -= q
        d[j] -= q
        if s[i] == 0:
            rows.remove(i)
        else:
            cols.remove(j)
    return alloc


def optimal(cost, supply, demand):
    m, n = len(supply), len(demand)
    c = np.array(cost).flatten()
    A, b = [], []
    for i in range(m):
        row = [0] * (m * n)
        for j in range(n):
            row[i * n + j] = 1
        A.append(row)
        b.append(supply[i])
    for j in range(n):
        row = [0] * (m * n)
        for i in range(m):
            row[i * n + j] = 1
        A.append(row)
        b.append(demand[j])
    result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not result.success:
        raise ValueError("The solver could not find a feasible solution.")
    return result.x.reshape(m, n).round(6).tolist()


METHODS = {
    "northwest": ("North-West Corner", northwest_corner),
    "least_cost": ("Least Cost", least_cost),
    "vogel": ("Vogel's Approximation (VAM)", vogel),
    "optimal": ("Optimal (linear programming)", optimal),
}


def total_cost(alloc, cost):
    return sum(alloc[i][j] * cost[i][j]
               for i in range(len(cost)) for j in range(len(cost[0])))


# ---------------------------------------------------------------
# TRANSPORTATION: validate -> balance -> solve -> package result
# ---------------------------------------------------------------
def solve_transportation(cost, supply, demand, method="all"):
    # ---- validate the input (errors go back to the AI so it can fix them) ----
    if not cost or any(len(row) != len(demand) for row in cost):
        raise ValueError("Every row of the cost matrix must have one value per destination.")
    if len(cost) != len(supply):
        raise ValueError("The cost matrix needs one row per source (supply value).")
    if any(x < 0 for x in supply) or any(x < 0 for x in demand):
        raise ValueError("Supply and demand values cannot be negative.")
    if method != "all" and method not in METHODS:
        raise ValueError("method must be one of: all, " + ", ".join(METHODS))

    cost = [list(map(float, r)) for r in cost]
    supply, demand = list(map(float, supply)), list(map(float, demand))
    m, n = len(supply), len(demand)
    row_names = [f"Source {i + 1}" for i in range(m)]
    col_names = [f"Destination {j + 1}" for j in range(n)]

    # ---- balance with a dummy row/column if needed ----
    note = "Problem was already balanced."
    if sum(supply) > sum(demand):
        for row in cost:
            row.append(0)
        demand.append(sum(supply) - sum(demand))
        col_names.append("Dummy destination")
        note = "Supply > demand, so a dummy destination (zero cost) was added."
    elif sum(demand) > sum(supply):
        cost.append([0] * len(demand))
        supply.append(sum(demand) - sum(supply))
        row_names.append("Dummy source")
        note = "Demand > supply, so a dummy source (zero cost) was added."

    # ---- run one method, or all of them ----
    chosen = list(METHODS) if method == "all" else [method]
    results = {}
    for key in chosen:
        label, func = METHODS[key]
        alloc = func(cost, supply, demand)
        shipments = [
            f"{row_names[i]} -> {col_names[j]}: {alloc[i][j]:g} units (cost {cost[i][j]:g} each)"
            for i in range(len(supply)) for j in range(len(demand)) if alloc[i][j] > 0
        ]
        results[label] = {"total_cost": round(total_cost(alloc, cost), 4),
                          "shipments": shipments}

    best = total_cost(optimal(cost, supply, demand), cost)
    return {"balancing": note, "results": results,
            "optimal_cost_for_reference": round(best, 4)}


# ---------------------------------------------------------------
# ASSIGNMENT
# ---------------------------------------------------------------
def solve_assignment(matrix, objective="min"):
    if not matrix or any(len(r) != len(matrix[0]) for r in matrix):
        raise ValueError("All rows of the matrix must have the same length.")
    if objective not in ("min", "max"):
        raise ValueError("objective must be 'min' or 'max'.")

    rows, cols = len(matrix), len(matrix[0])
    size = max(rows, cols)
    square = np.zeros((size, size))
    square[:rows, :cols] = np.array(matrix, dtype=float)

    work = square.max() - square if objective == "max" else square
    worker_ids, job_ids = linear_sum_assignment(work)

    assignments, total = [], 0.0
    for w, j in zip(worker_ids, job_ids):
        if w < rows and j < cols:
            assignments.append(f"Worker/row {w + 1} -> Job/column {j + 1} (value {square[w][j]:g})")
            total += square[w][j]
    note = "Matrix was square."
    if rows != cols:
        note = "Matrix was not square, so dummy rows/columns of zeros were added (they are ignored in the output)."
    return {"note": note, "objective": objective,
            "assignments": assignments, "total_value": round(float(total), 4)}