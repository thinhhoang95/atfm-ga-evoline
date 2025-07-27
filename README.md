# MOMMA Toy Implementation

This repository provides a small-scale reconstruction of the Multi-Objective Multi-Memetic Algorithm (MOMMA) inspired by Yan & Cai (2017). The code focuses on the algorithmic core and demonstrates it on a minimal air-traffic scenario with three flights and two sectors.

## Requirements

The Python dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Running

Execute the toy example with:

```bash
python3 momma_toy.py
```

The script runs a short NSGA-II optimisation and prints the Pareto optimal individuals along with their objective values `(conflicts, cost, sector violations)`.
