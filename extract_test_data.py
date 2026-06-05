import csv
import sqlite3
import os

# The airports we want to isolate
TARGET_AIRPORTS = {"AUS", "DFW", "IAH", "SAT", "OKC"}

print("Starting data extraction...")

# 1. Filter metro_areas.csv and collect matching names/IDs
target_names = set()
target_ids = set()

with open("PERF/metro_areas.csv", "r", encoding="utf-8") as fin, \
     open("PERF/metro_areas_test.csv", "w", newline="", encoding="utf-8") as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        airport_code = row[4].strip('"')
        if airport_code in TARGET_AIRPORTS:
            writer.writerow(row)
            target_ids.add(row[0].strip('"'))
            target_names.add(row[1].strip('"'))

print(f"Isolated metros: {target_names}")

# 2. Filter served_from.csv
with open("SERVEDFROM_DATA/served_from.csv", "r", encoding="utf-8") as fin, \
     open("SERVEDFROM_DATA/served_from_test.csv", "w", newline="", encoding="utf-8") as fout:
    
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        asn_metro = row[0].strip('"')
        bw_metro = row[1].strip('"')
        # Only keep traffic strictly between our target metros
        if asn_metro in target_names and bw_metro in target_names:
            writer.writerow(row)

print("Filtered CSVs successfully.")

# 3. Filter the SQLite database
db_path = "PERF/perf_data.db"
test_db_path = "PERF/perf_data_test.db"

if os.path.exists(test_db_path):
    os.remove(test_db_path)

conn = sqlite3.connect(db_path)
# Attach the new database so we can copy directly over
conn.execute(f"ATTACH DATABASE '{test_db_path}' AS test_db")

# Helper function to copy table schema and filter data
def filter_table(table_name, where_clause):
    print(f"Filtering {table_name}...")
    # Create an identical empty table in the test database
    conn.execute(f"CREATE TABLE test_db.{table_name} AS SELECT * FROM {table_name} WHERE 0")
    # Copy filtered data
    conn.execute(f"INSERT INTO test_db.{table_name} SELECT * FROM {table_name} WHERE {where_clause}")

# Format our sets for SQL IN clauses
names_sql = "(" + ",".join([f"'{n}'" for n in target_names]) + ")"
ids_sql = "(" + ",".join([f"'{i}'" for i in target_ids]) + ")"

# Filter the 4 core performance tables based on how analyse.py queries them
filter_table("netopt_perf_edge_rtt_ansabni", f"region_metro IN {names_sql} AND client_metro IN {ids_sql}")
filter_table("netopt_perf_edge_ecor_tat_ansabni", f"edge_metro IN {names_sql}")
filter_table("netopt_perf_midgress_rtt_ansabni", f"parent_metro IN {names_sql} AND child_metro IN {names_sql}")
filter_table("netopt_perf_midgress_ecor_tat_ansabni", f"edge_metro IN {names_sql}")

conn.commit()
conn.close()

print(f"Database extraction complete. Saved to {test_db_path}")