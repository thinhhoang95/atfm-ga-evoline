#!/usr/bin/env python3
"""Toy implementation of the MOMMA algorithm for a minimal air traffic example."""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from deap import base, creator, tools, algorithms
from shapely.geometry import Point, Polygon

# discretisation constants
TIME_SLOT = 20  # seconds
WINDOW_SLOTS = 15  # 5 minutes
LEVELS = [300, 320]

@dataclass
class Node:
    id: str
    lat: float
    lon: float

@dataclass
class Sector:
    id: str
    polygon: Polygon
    lvl_min: int
    lvl_max: int
    capacity: int

@dataclass
class Flight:
    id: str
    route: List[str]           # node ids
    tta: List[int]             # times in slots
    levels: List[int]
    dep: int
    arr: int
    route_nodes_f: List[str] = None
    t_entry_f: Dict[int, int] = None
    t_exit_f: Dict[int, int] = None
    level_f: Dict[int, int] = None
    kc_f: int = -1
    hf_k: List[int] = None
    limits: Dict = None

# ---------------------------------------------------------------------------
# Toy network definition
# ---------------------------------------------------------------------------

# nodes placed on a grid for simplicity
nodes = {
    "A": Node("A", 0, 0),
    "B": Node("B", 1, 0),
    "C": Node("C", 2, 0),
    "D": Node("D", 0, 1),
    "E": Node("E", 1, 1),
    "F": Node("F", 2, 1),
}

# sectors as horizontal rectangles
sector1 = Sector("S1", Polygon([(-1, -1), (3, -1), (3, 0.5), (-1, 0.5)]), 300, 320, 2)
sector2 = Sector("S2", Polygon([(-1, 0.5), (3, 0.5), (3, 2), (-1, 2)]), 300, 320, 2)
sectors = [sector1, sector2]

# flights with nominal times (slots) at each node
flights = {
    "F1": Flight("F1", ["A", "B", "C"], [0, 5, 10], [300, 300, 300], dep=0, arr=10),
    "F2": Flight("F2", ["D", "E", "F"], [0, 5, 10], [300, 300, 300], dep=6, arr=16),
    "F3": Flight("F3", ["A", "B", "E", "F"], [0, 5, 10, 15], [300, 300, 300, 300], dep=15, arr=30),
}

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

# map node and level to conflict region index
CR_MAP: Dict[tuple, int] = {}
for node_id, node in nodes.items():
    p = Point(node.lat, node.lon)
    for cr_idx, cr in enumerate(sectors):
        if cr.polygon.contains(p):
            for lvl in LEVELS:
                if cr.lvl_min <= lvl <= cr.lvl_max:
                    CR_MAP[(node_id, lvl)] = cr_idx

# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def compute_flight_state(flight: Flight, delay: int = 0, level_change: Dict[int, int] = None):
    """Return times, levels and sectors for a flight with modifications."""
    if level_change is None:
        level_change = {}
    times = [t + delay for t in flight.tta]
    levels = flight.levels.copy()
    for idx, lvl in level_change.items():
        if 0 <= idx < len(levels):
            levels[idx] = lvl
    sectors_per_leg = [CR_MAP.get((n, l), -1) for n, l in zip(flight.route, levels)]
    return times, levels, sectors_per_leg


def compute_TI(state) -> tuple[dict, dict]:
    """
    return ϕ (nested dict) and U (dict flight-id → float)
    state[fid] = (times[], levels[], sectors[])
    """
    phi = {fid: {} for fid in state}
    U = {fid: 0.0 for fid in state}

    # Simplified x_tkl representation: flight presence in a sector at a time slot
    x = {}
    for fid, (times, levels, crs) in state.items():
        for t, cr in zip(times, crs):
            if cr != -1:
                x.setdefault((t, cr), set()).add(fid)

    for f_i, (times_i, _, crs_i) in state.items():
        # Denominator: total presence of f_i in all sectors
        total_presence_i = sum(1 for cr in crs_i if cr != -1)
        if total_presence_i == 0:
            continue

        for f_j, (times_j, _, crs_j) in state.items():
            if f_i == f_j:
                continue

            # Numerator: co-location count
            overlap = 0
            for t_i, cr_i in zip(times_i, crs_i):
                if cr_i != -1:
                    # Check if f_j is in the same sector at the same time
                    if x.get((t_i, cr_i)) and f_j in x[(t_i, cr_i)]:
                        overlap += 1

            if total_presence_i > 0:
                phi[f_i][f_j] = overlap / total_presence_i
            else:
                phi[f_i][f_j] = 0

    for f_i in state:
        U[f_i] = sum(phi[f_i].get(f_j, 0) for f_j in state if f_i != f_j)

    return phi, U


def ttls(individual, fid, state, U, T_GH=5):
    """Time-Tuning Local Search."""
    flight = flights[fid]
    times, _, crs = state[fid]

    # Determine most conflicting CR
    if not any(c != -1 for c in crs):
        return individual  # No CRs for this flight

    cr_conflicts = {}
    for cr_idx in set(c for c in crs if c != -1):
        # Simplified: count overlaps in this CR
        # A proper implementation would use the x_tkl structure from compute_TI
        conflicts = 0
        for i, (t, cr) in enumerate(zip(times, crs)):
            if cr == cr_idx:
                for other_fid, (other_times, _, other_crs) in state.items():
                    if fid == other_fid:
                        continue
                    for j, (other_t, other_cr) in enumerate(zip(other_times, other_crs)):
                        if other_cr == cr_idx and other_t == t:
                            conflicts += 1
        cr_conflicts[cr_idx] = conflicts

    if not cr_conflicts:
        return individual # No conflicts involving this flight

    kc_f = max(cr_conflicts, key=cr_conflicts.get)

    # Simplified peak-slot and direction decision
    # (A full implementation is more complex)

    # For this toy version, we'll just try a small random shift
    current_delay = individual['delay_d'].get(fid, 0)

    # Decide direction (simplified)
    # A real implementation would analyze peak conflict time vs entry/exit
    shift_direction = np.random.choice([-1, 1])

    best_shift = 0
    min_conflicts = float('inf')

    # Explore shifts
    for shift in range(1, T_GH + 1):
        t = shift_direction * shift

        # Create a temporary state with the new delay
        temp_individual = individual.copy()
        temp_individual['delay_d'] = individual['delay_d'].copy()
        temp_individual['delay_d'][fid] = current_delay + t

        temp_state = {}
        for f_id, fl in flights.items():
            temp_state[f_id] = compute_flight_state(
                fl,
                temp_individual['delay_d'].get(f_id, 0),
                individual['levels'].get(f_id, {})
            )

        # Evaluate conflicts in the most conflicting CR (kc_f)
        conflicts = 0
        new_times, _, new_crs = temp_state[fid]
        for i, (time, cr) in enumerate(zip(new_times, new_crs)):
            if cr == kc_f:
                for other_fid, (other_times, _, other_crs) in temp_state.items():
                    if fid == other_fid:
                        continue
                    for j, (other_t, other_cr) in enumerate(zip(other_times, other_crs)):
                        if other_cr == kc_f and other_t == time:
                            conflicts += 1

        if conflicts < min_conflicts:
            min_conflicts = conflicts
            best_shift = t

    # Apply the best shift found
    final_delay = current_delay + best_shift
    individual['delay_d'][fid] = final_delay

    # Update TTA for the modified flight
    individual['TTA'][fid] = [t + final_delay for t in flight.tta]

    return individual


def lals(individual, fid, state, U, N_FLC=1):
    """Level-Adjustment Local Search."""
    flight = flights[fid]
    times, levels, crs = state[fid]

    current_level_changes = individual['levels'].get(fid, {})
    if len(current_level_changes) >= N_FLC:
        return individual  # Max level changes reached

    # Determine most conflicting CR (reusing logic from TTLS for simplicity)
    if not any(c != -1 for c in crs):
        return individual

    cr_conflicts = {}
    for cr_idx in set(c for c in crs if c != -1):
        conflicts = 0
        for i, (t, cr) in enumerate(zip(times, crs)):
            if cr == cr_idx:
                for other_fid, (other_times, _, other_crs) in state.items():
                    if fid == other_fid:
                        continue
                    for j, (other_t, other_cr) in enumerate(zip(other_times, other_crs)):
                        if other_cr == cr_idx and other_t == t:
                            conflicts += 1
        cr_conflicts[cr_idx] = conflicts

    if not cr_conflicts:
        return individual

    kc_f = max(cr_conflicts, key=cr_conflicts.get)

    # Find index of the first node in the most conflicting CR
    kc_f_node_idx = -1
    for i, cr in enumerate(crs):
        if cr == kc_f:
            kc_f_node_idx = i
            break

    if kc_f_node_idx == -1:
        return individual

    # Try changing level up or down
    best_level_change = {}
    min_conflicts = float('inf')

    for delta in [-1, 1]:  # Try delta = -1 and +1
        new_level = levels[kc_f_node_idx] + delta * 20  # Levels are in steps of 20
        if not (min(LEVELS) <= new_level <= max(LEVELS)):
            continue

        # Create a temporary individual with the new level change
        temp_individual = individual.copy()
        temp_individual['levels'] = individual['levels'].copy()
        temp_individual['levels'][fid] = individual['levels'].get(fid, {}).copy()

        # Apply level change from kc_f onwards
        for i in range(kc_f_node_idx, len(flight.route)):
            temp_individual['levels'][fid][i] = new_level

        # Create temporary state
        temp_state = {}
        for f_id, fl in flights.items():
            temp_state[f_id] = compute_flight_state(
                fl,
                individual['delay_d'].get(f_id, 0),
                temp_individual['levels'].get(f_id, {})
            )

        # Evaluate conflicts
        conflicts = 0
        # This is a simplification. A full eval would be needed.
        for i in range(len(state)):
            ti, _, _ = temp_state[list(state.keys())[i]]
            for j in range(i + 1, len(state)):
                tj, _, _ = temp_state[list(state.keys())[j]]
                # Simplified conflict check
                if any(t1 == t2 for t1 in ti for t2 in tj):
                     conflicts +=1

        if conflicts < min_conflicts:
            min_conflicts = conflicts
            # This is a simplified application rule
            best_level_change = {i: new_level for i in range(kc_f_node_idx, len(flight.route))}

    # Apply the best level change
    if best_level_change:
        individual.setdefault('levels', {})
        individual['levels'].setdefault(fid, {}).update(best_level_change)

    return individual


def evaluate(individual):
    """Compute objectives (conflicts, cost, cr violations)."""
    delays = individual.get('delay_d', {})
    level_changes = individual.get('levels', {})

    flight_state = {}
    for fid, fl in flights.items():
        flight_state[fid] = compute_flight_state(
            fl,
            delays.get(fid, 0),
            level_changes.get(fid, {})
        )

    # Obj1: conflicts (simplified check at same node/time/level)
    conflict = 0
    ids = list(flight_state.keys())
    for i in range(len(ids)):
        ti, li, _ = flight_state[ids[i]]
        for j in range(i+1, len(ids)):
            tj, lj, _ = flight_state[ids[j]]
            for n_i, t_i, l_i in zip(flights[ids[i]].route, ti, li):
                for n_j, t_j, l_j in zip(flights[ids[j]].route, tj, lj):
                    if n_i == n_j and t_i == t_j and l_i == l_j:
                        conflict += 1

    # Obj2: modification cost (delay + number of level changes)
    cost = sum(delays.values()) + sum(len(v) for v in level_changes.values())

    # Obj3: cr overload violations per window
    max_windows = (max(fl.arr for fl in flights.values()) // WINDOW_SLOTS) + 1
    window_occ = {(w, s): 0 for w in range(max_windows) for s in range(len(sectors))}
    for times, lvls, sec in flight_state.values():
        for t, s_idx in zip(times, sec):
            if s_idx < 0:
                continue
            w = t // WINDOW_SLOTS
            if (w, s_idx) in window_occ:
                window_occ[(w, s_idx)] += 1
    viol = 0
    for (w, s_idx), occ in window_occ.items():
        if s_idx >= 0:
            cap = sectors[s_idx].capacity
            if occ > cap:
                viol += occ - cap

    return conflict, cost, viol

# ---------------------------------------------------------------------------
# DEAP setup
# ---------------------------------------------------------------------------
creator.create('FitnessMulti', base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create('Individual', dict, fitness=creator.FitnessMulti)

def init_individual():
    individual = creator.Individual({
        "delay_d": {},
        "TTA": {},
        "levels": {},
    })
    return individual

toolbox = base.Toolbox()

toolbox.register('individual', init_individual)

toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', evaluate)

def cx(ind1, ind2):
    if np.random.rand() < 0.5:
        ind1['delay_d'], ind2['delay_d'] = ind2['delay_d'], ind1['delay_d']
    else:
        ind1['levels'], ind2['levels'] = ind2['levels'], ind1['levels']
    return ind1, ind2

def mut(ind, U_Th=0.015, k_intervals=None):
    """Mutation operator that applies RLS, TTLS, or LALS."""
    if k_intervals is None:
        k_intervals = {
            'RLS': (0, 0.33),
            'TTLS': (0.33, 0.66),
            'LALS': (0.66, 1.0)
        }

    # First, calculate the current state and U-vector for the individual
    current_state = {}
    for fid, fl in flights.items():
        current_state[fid] = compute_flight_state(
            fl,
            ind['delay_d'].get(fid, 0),
            ind['levels'].get(fid, {})
        )
    _, U = compute_TI(current_state)

    # Determine which operator to use based on a random k
    k = np.random.rand()
    operator = None
    if k_intervals['RLS'][0] <= k < k_intervals['RLS'][1]:
        operator = 'RLS'
    elif k_intervals['TTLS'][0] <= k < k_intervals['TTLS'][1]:
        operator = 'TTLS'
    elif k_intervals['LALS'][0] <= k < k_intervals['LALS'][1]:
        operator = 'LALS'

    # Select candidate flights
    candidates = {f for f, u_val in U.items() if u_val > U_Th}
    if not candidates:
        return ind,  # No flights to modify

    # Choose a flight to modify
    fid_to_modify = np.random.choice(list(candidates))

    # Apply the chosen local search operator
    original_fitness = ind.fitness.values

    if operator == 'TTLS':
        # TTLS returns a new individual, so we work with a copy
        new_ind = ttls(ind.copy(), fid_to_modify, current_state, U)
    elif operator == 'LALS':
        new_ind = lals(ind.copy(), fid_to_modify, current_state, U)
    else: # RLS or fallback
        # RLS is the original mutation logic, simplified here
        if np.random.rand() < 0.5:
            d = ind['delay_d'].get(fid_to_modify, 0) + np.random.randint(-1, 2)
            ind['delay_d'][fid_to_modify] = max(0, d)
        else:
            idx = np.random.randint(0, len(flights[fid_to_modify].route))
            lvl = np.random.choice(LEVELS)
            ind.setdefault('levels', {})
            ind.setdefault(fid_to_modify, {})[idx] = lvl
        new_ind = ind

    # Acceptance criteria (simplified)
    # A full implementation would use the weighted fitness function
    new_fitness = evaluate(new_ind)

    # Simple dominance check
    if tools.emo.isDominated(original_fitness, new_fitness):
         # The new individual is worse, so we might reject it
         # For simplicity in this toy example, we'll accept it to see changes
         pass

    # In a real scenario, you'd handle acceptance more carefully
    # For now, we just return the potentially modified individual
    final_ind = creator.Individual(new_ind)
    return final_ind,

toolbox.register('mate', cx)

toolbox.register('mutate', mut)

toolbox.register('select', tools.selNSGA2)

# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------

def main():
    np.random.seed(0)
    pop = toolbox.population(n=20)
    ngen = 40
    mu = 20
    lam = 20
    cxpb = 0.9
    mutpb = 0.1

    algorithms.eaMuPlusLambda(
        pop, toolbox, mu=mu, lambda_=lam, cxpb=cxpb, mutpb=mutpb,
        ngen=ngen
    )

    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    for ind in pareto:
        print('Ind:', ind, 'fitness:', ind.fitness.values)
    return pareto

if __name__ == '__main__':
    main()
