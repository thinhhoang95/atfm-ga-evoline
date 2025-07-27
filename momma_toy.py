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

# map node and level to sector index
SECTOR_MAP: Dict[tuple, int] = {}
for node_id, node in nodes.items():
    p = Point(node.lat, node.lon)
    for s_idx, sec in enumerate(sectors):
        if sec.polygon.contains(p):
            for lvl in LEVELS:
                if sec.lvl_min <= lvl <= sec.lvl_max:
                    SECTOR_MAP[(node_id, lvl)] = s_idx

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
    sectors_per_leg = [SECTOR_MAP.get((n, l), -1) for n, l in zip(flight.route, levels)]
    return times, levels, sectors_per_leg


def evaluate(individual):
    """Compute objectives (conflicts, cost, sector violations)."""
    delays = individual.get('delays', {})
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

    # Obj3: sector overload violations per window
    window_occ = {(w, s): 0 for w in range(4) for s in range(len(sectors))}
    for times, lvls, sec in flight_state.values():
        for t, s_idx in zip(times, sec):
            if s_idx < 0:
                continue
            w = t // WINDOW_SLOTS
            window_occ[(w, s_idx)] += 1
    viol = 0
    for (w, s_idx), occ in window_occ.items():
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
    return creator.Individual({'delays': {}, 'levels': {}})

toolbox = base.Toolbox()

toolbox.register('individual', init_individual)

toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('evaluate', evaluate)

def cx(ind1, ind2):
    if np.random.rand() < 0.5:
        ind1['delays'], ind2['delays'] = ind2['delays'], ind1['delays']
    else:
        ind1['levels'], ind2['levels'] = ind2['levels'], ind1['levels']
    return ind1, ind2

def mut(ind):
    fid = np.random.choice(list(flights.keys()))
    if np.random.rand() < 0.5:
        d = ind['delays'].get(fid, 0) + np.random.randint(-1, 2)
        ind['delays'][fid] = max(0, d)
    else:
        idx = np.random.randint(0, len(flights[fid].route))
        lvl = np.random.choice(LEVELS)
        ind.setdefault('levels', {})
        ind['levels'].setdefault(fid, {})[idx] = lvl
    return ind,

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
        ngen=ngen, verbose=False
    )

    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    for ind in pareto:
        print('Ind:', ind, 'fitness:', ind.fitness.values)
    return pareto

if __name__ == '__main__':
    main()
