"""Quick test to diagnose effcaps.csv parsing."""
import csv
from pathlib import Path

effcap_path = Path(__file__).resolve().parent / "COST" / "effcaps.csv"

print(f"File exists: {effcap_path.exists()}")
print()

# Show raw first two lines
with effcap_path.open(encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(f"Raw line {i}: {repr(line)}")
        if i >= 2:
            break

print()

# Show what DictReader sees as fieldnames BEFORE reading any rows
with effcap_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print(f"fieldnames before first read: {reader.fieldnames!r}")
    first_row = next(iter(reader))
    print(f"fieldnames after first read:  {reader.fieldnames!r}")
    print(f"first row keys: {list(first_row.keys())!r}")
    print(f"first row values: {list(first_row.values())!r}")

print()

# The fix: use utf-8-sig which automatically strips the BOM
with effcap_path.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    print(f"Fixed fieldnames: {reader.fieldnames!r}")
    for row in reader:
        key = (row["macroarea"].strip(), row["traffic_class"].strip())
        val = float(row["effcap"].strip())
        print(f"  key={key}  effcap={val:.1f}")
        break  # just show first row
print("\nParsing succeeded!")
