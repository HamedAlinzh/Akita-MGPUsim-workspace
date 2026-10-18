#!/usr/bin/env python3
"""Summarize tile-completion spread from the tile tracker's CSVs.

A completion group ends where the gap to the next completion exceeds 5% of
the span from first to last completion and 1 us.

Usage: python3 scripts/tile_spread.py results/step1_tracker/tiles_*.csv
"""
import csv
import sys


def summarize(path):
    c = sorted(float(r["complete_us"]) for r in csv.DictReader(open(path)))
    span = c[-1] - c[0]
    groups = 1 + sum(1 for a, b in zip(c, c[1:]) if b - a > max(0.05 * span, 1.0))
    return len(c), c[0], c[len(c) // 2], c[-1], groups


print("csv | tiles | first_us | median_us | last_us | first/last | groups")
for path in sys.argv[1:]:
    n, first, med, last, groups = summarize(path)
    print(f"{path.rsplit('/', 1)[-1]} | {n} | {first:.1f} | {med:.1f} | {last:.1f} | "
          f"{first / last:.2f} | {groups}")
