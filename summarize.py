import re
from pathlib import Path

def parse_last_iteration(filepath):
    """Return a dict of metro -> disk_mb from the last iteration block."""
    text = Path(filepath).read_text(encoding='utf-8')
    # Split on iteration= lines
    blocks = re.split(r'\niteration=\d+\n', '\n' + text)
    # Last non-empty block
    for block in reversed(blocks):
        block = block.strip()
        if block:
            break
    result = {}
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(\w+),\s*disk_mb=([\d.]+)', line)
        if m:
            result[m.group(1)] = float(m.group(2))
    return result

f1 = 'US_traffic_based_penalty/gradient_descent_metro_log.txt'
f2 = 'US_aggressive_perf_penalty/gradient_descent_metro_log.txt'

d1 = parse_last_iteration(f1)
d2 = parse_last_iteration(f2)

all_metros = sorted(set(d1) | set(d2))

# Header
print(f"{'Metro':<8} {'Traffic-based penalty (TB)':>28} {'Aggressive perf penalty (TB)':>30}")
print('-' * 70)
for metro in all_metros:
    v1 = d1.get(metro, float('nan'))
    v2 = d2.get(metro, float('nan'))
    tb1 = f"{v1/1_000_000:.2f}" if v1 == v1 else 'N/A'
    tb2 = f"{v2/1_000_000:.2f}" if v2 == v2 else 'N/A'
    print(f"{metro:<8} {tb1:>28} {tb2:>30}")