import csv
import sys

import sqlite3
from probability import ProbabilityDensityFunction

try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

client_metro_ids = {}
client_metro_ids['Seattle'] = 1178
client_metro_ids['San_Jose'] = 1176
client_metro_ids['Los_Angeles'] = 1155
client_metro_ids['Mexico_City'] = 1113
client_metro_ids['Dallas'] = 1147
client_metro_ids['Vancouver'] = 1027
client_metro_ids['Hyderabad'] = 1094
client_metro_ids['Bangalore'] = 1090
client_metro_ids['Bogota'] = 1052
client_metro_ids['Pittsburgh'] = 1166
client_metro_ids['Salt_Lake_City'] = 1173
client_metro_ids['Phoenix'] = 1165


client_metros = {}
client_metros[1178] = 'Seattle'
client_metros[1176] = 'San_Jose'
client_metros[1155] = 'Los_Angeles'
client_metros[1113] = 'Mexico_City'
client_metros[1147] = 'Dallas'
client_metros[1027] = 'Vancouver'
client_metros[1094] = 'Hyderabad'
client_metros[1090] = 'Bangalore'
client_metros[1052] = 'Bogota'
client_metros[1166] = 'Pittsburgh'
client_metros[1173] = 'Salt_Lake_City'
client_metros[1165] = 'Phoenix'

# I have a file in PERF/metro_areas.csv. Ignore the first line, and then use the first two columns to find the metro id and metroname mapping. Populate the client_metro_ids and client_metros dictionaries with this mapping. The first column is metro id and the second column is metro name.

with open("PERF/metro_areas.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        metro_id = int(row[0])
        metro_name = row[1] 
        client_metro_ids[metro_name] = metro_id
        client_metros[metro_id] = metro_name

# I have a database in PERF/perf_data.db that is used by MetroPerformance.
# Let us first analyse the rtt distributions

# The database has a table called netopt_perf_edge_rtt_ansabni
# It has the following columns: index|pdate|timebin|bucket|client_metro|region_metro|total_requests|total_bytes_sent|rtt_0_5_ms|rtt_5_10_ms|rtt_10_15_ms|rtt_15_20_ms|rtt_20_25_ms|
# rtt_25_30_ms|rtt_30_35_ms|rtt_35_40_ms|rtt_40_45_ms|rtt_45_50_ms|rtt_50_55_ms|rtt_55_60_ms|rtt_60_65_ms|rtt_65_70_ms|rtt_70_75_ms|rtt_80_85_ms|rtt_85_90_ms|rtt_90_95_ms|rtt_95_100_ms|
# rtt_100_105_ms|rtt_105_110_ms|rtt_110_115_ms|rtt_115_120_ms|rtt_120_125_ms|rtt_125_130_ms|rtt_130_135_ms|rtt_135_140_ms|rtt_140_145_ms|rtt_145_150_ms|rtt_150_155_ms|rtt_155_160_ms|
# rtt_160_165_ms|rtt_165_170_ms|rtt_170_175_ms|rtt_175_180_ms|rtt_180_185_ms|rtt_185_190_ms|rtt_190_195_ms|rtt_195_200_ms|rtt_200_205_ms|rtt_205_210_ms|rtt_210_215_ms|rtt_215_220_ms|
# rtt_220_225_ms|rtt_225_230_ms|rtt_230_235_ms|rtt_235_240_ms|rtt_240_245_ms|rtt_245_250_ms|rtt_250_255_ms|rtt_255_260_ms|rtt_260_265_ms|rtt_265_270_ms|rtt_270_275_ms|rtt_275_280_ms|
# rtt_280_285_ms|rtt_285_290_ms|rtt_290_295_ms|rtt_295_300_ms|rtt_300_inf_ms

#write me a function that takes metro_name as an argument and finds the top 5 metros that send requests to that metro, based on the total_requests column in the netopt_perf_edge_rtt_ansabni table. 
# The function should return a list of tuples, where each tuple contains the client metro name and the total requests from that metro to the given metro.

def get_top_sending_metros(metro_name: str) -> list[tuple[str, int]]:
    """Return the top 5 metros sending requests to ``metro_name``."""
    query = f"""
    SELECT client_metro, SUM(total_requests) as total_requests
    FROM netopt_perf_edge_rtt_ansabni
    WHERE region_metro = '{metro_name}' AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')
    GROUP BY client_metro
    ORDER BY total_requests DESC
    LIMIT 10
    """

    import pandas as pd

    conn = sqlite3.connect("PERF/perf_data.db")
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        return []

    return list(zip(df["client_metro"], df["total_requests"]))

'''
i = 1
for metro_name in client_metro_ids:
    top_metros = get_top_sending_metros(metro_name)
    print(f"Top metros sending requests to {metro_name}:")
    for client_metro_id, total_requests in top_metros:
        client_metro_name = client_metros.get(client_metro_id, f"Unknown({client_metro_id})")
        print(f"  {client_metro_name} ({client_metro_id}): {total_requests} requests")

    i += 1
    if i > 12:
        break
'''

'''
for metro_name in ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Dhaka", "Kolkata"]:
    top_metros = get_top_sending_metros(metro_name)
    print(f"Top metros sending requests to {metro_name}:")
    for client_metro_id, total_requests in top_metros:
        client_metro_name = client_metros.get(client_metro_id, f"Unknown({client_metro_id})")
        print(f"  {client_metro_name} ({client_metro_id}): {total_requests} requests")
        
import sys
sys.exit()
'''


# Write me a function that takes two arguments: metro_name and client_metro_id
# The function should connect to the database PERF/perf_data.db, query the table netopt_perf_edge_rtt_ansabni
# for the given metro_name and client_metro_id, and return a ProbabilityDensityFunction
# representing the RTT distribution for that metro and client metro.


def get_rtt_pdf(metro_name: str, client_metro_id: int) -> ProbabilityDensityFunction:
    """Return an RTT PDF for ``metro_name`` and ``client_metro_id``."""

    import pandas as pd

    query = f"""
    SELECT * FROM netopt_perf_edge_rtt_ansabni
    WHERE region_metro = '{metro_name}' AND client_metro = '{client_metro_id}'
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""

    df = None
    conn = sqlite3.connect("PERF/perf_data.db")
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying database: {e}")
        return ProbabilityDensityFunction([])
    finally:
        conn.close()

    if df is None or df.empty:
        print(f"No data found for metro {metro_name} and client metro {client_metro_id}")
        return ProbabilityDensityFunction([])
    
    pdf = ProbabilityDensityFunction.from_dataframe(df)

    return pdf

# Get RTT PDFs for each metro and plot them
# I dont want it to show me the plots, just save them to files
'''
for metro_name in client_metro_ids:
    for client_metro_id in client_metros:
        rtt_pdf = get_rtt_pdf(metro_name, client_metro_id)
        if plt is not None:
            rtt_pdf.plot(title=f"RTT PDF for {metro_name} from clients in {client_metros[client_metro_id]}")
            plt.savefig(f"rtts/rtt_pdf_{metro_name}_from_{client_metros[client_metro_id]}.png")
            plt.clf()
'''

# Okay, so now lets analyse the TATs for cache hits at the edge
# We have the table netopt_perf_edge_ecor_tat_ansabni it has the following columns
# index|pdate|timebin|edge_metro|cache_hit_type|total_requests|total_bytes_sent|tat_0_2_ms|tat_2_4_ms|tat_4_6_ms|tat_6_8_ms|tat_8_10_ms|
# tat_10_12_ms|tat_12_14_ms|tat_14_16_ms|tat_16_18_ms|tat_18_20_ms|tat_20_22_ms|tat_22_24_ms|tat_24_26_ms|tat_26_28_ms|tat_28_30_ms|tat_30_35_ms|
# tat_35_40_ms|tat_40_45_ms|tat_45_50_ms|tat_50_55_ms|tat_55_60_ms|tat_60_65_ms|tat_65_70_ms|tat_70_75_ms|tat_75_80_ms|tat_80_85_ms|tat_85_90_ms|tat_90_95_ms|tat_95_100_ms|tat_100_inf_ms
# Here cache_hit_type: 0 means cache_miss, 1 means cache_hit, 2 means icp_cache_hit
# Okay, so let us write a function that gets the TAT PDF for edge cache hits for a given metro

def get_edge_tat_pdf(metro_name: str, cache_hit_type: int = 1) -> ProbabilityDensityFunction:
    """Return an edge TAT PDF for ``metro_name``."""
    import pandas as pd

    query = f"""
    SELECT * FROM netopt_perf_edge_ecor_tat_ansabni
    WHERE edge_metro = '{metro_name}' AND cache_hit_type = {cache_hit_type}
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""

    df = None
    conn = sqlite3.connect("PERF/perf_data.db")
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying database: {e}")
        return ProbabilityDensityFunction([])
    finally:
        conn.close()

    if df is None or df.empty:
        print(f"No data found for metro {metro_name} edge TATs")
        return ProbabilityDensityFunction([])
    
    pdf = ProbabilityDensityFunction.from_dataframe(df)
    return pdf
'''
# Get edge TAT PDFs for each metro and plot them
for metro_name in client_metro_ids:
    edge_tat_pdf = get_edge_tat_pdf(metro_name)
    if plt is not None:
        edge_tat_pdf.plot(title=f"Edge TAT PDF for {metro_name} (Cache Hits)")
        plt.savefig(f"tats_edge/edge_tat_pdf_{metro_name}_cache_hits.png")
        plt.clf()
'''

# Now similarly, I have rtts between every pair of metros in the table netopt_perf_midgress_rtt_ansabni
# Here are the columns: index|pdate|timebin|parent_metro|child_metro|total_requests|total_bytes_sent|rtt_0_5_ms|rtt_5_10_ms|rtt_10_15_ms|
# rtt_15_20_ms|rtt_20_25_ms|rtt_25_30_ms|rtt_30_35_ms|rtt_35_40_ms|rtt_40_45_ms|rtt_45_50_ms|rtt_50_55_ms|rtt_55_60_ms|rtt_60_65_ms|
# rtt_65_70_ms|rtt_70_75_ms|rtt_80_85_ms|rtt_85_90_ms|rtt_90_95_ms|rtt_95_100_ms|rtt_100_105_ms|rtt_105_110_ms|rtt_110_115_ms|rtt_115_120_ms|
# rtt_120_125_ms|rtt_125_130_ms|rtt_130_135_ms|rtt_135_140_ms|rtt_140_145_ms|rtt_145_150_ms|rtt_150_155_ms|rtt_155_160_ms|rtt_160_165_ms|
# rtt_165_170_ms|rtt_170_175_ms|rtt_175_180_ms|rtt_180_185_ms|rtt_185_190_ms|rtt_190_195_ms|rtt_195_200_ms|rtt_200_205_ms|rtt_205_210_ms|
# rtt_210_215_ms|rtt_215_220_ms|rtt_220_225_ms|rtt_225_230_ms|rtt_230_235_ms|rtt_235_240_ms|rtt_240_245_ms|rtt_245_250_ms|rtt_250_255_ms|
# rtt_255_260_ms|rtt_260_265_ms|rtt_265_270_ms|rtt_270_275_ms|rtt_275_280_ms|rtt_280_285_ms|rtt_285_290_ms|rtt_290_295_ms|rtt_295_300_ms|rtt_300_inf_ms
# Let us write a function that gets the midgress RTT PDF between two metros

def get_midgress_rtt_pdf(parent_metro: str, child_metro: str) -> ProbabilityDensityFunction:
    """Return a midgress RTT PDF between ``parent_metro`` and ``child_metro``."""
    import pandas as pd

    query = f"""
    SELECT * FROM netopt_perf_midgress_rtt_ansabni
    WHERE parent_metro = '{parent_metro}' AND child_metro = '{child_metro}'
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""

    df = None
    conn = sqlite3.connect("PERF/perf_data.db")
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying database: {e}")
        return ProbabilityDensityFunction([])
    finally:
        conn.close()

    if df is None or df.empty:
        print(f"No data found for midgress RTT between {parent_metro} and {child_metro}")
        return ProbabilityDensityFunction([])
    
    pdf = ProbabilityDensityFunction.from_dataframe(df)
    return pdf
'''
# Get midgress RTT PDFs for each pair of metros and plot them
for parent_metro in client_metro_ids:
    for child_metro in client_metro_ids:
        midgress_rtt_pdf = get_midgress_rtt_pdf(parent_metro, child_metro)
        if plt is not None:
            midgress_rtt_pdf.plot(title=f"Midgress RTT PDF from {parent_metro} to {child_metro}")
            plt.savefig(f"rtts_midgress/midgress_rtt_pdf_{parent_metro}_to_{child_metro}.png")
            plt.clf()
'''

# Now lets analyse the TATs at the parent metro for cache hits and misses both
# table name: netopt_perf_midgress_ecor_tat_ansabni
# Columns: index|pdate|timebin|edge_metro|cache_hit_type|total_requests|total_bytes_sent|tat_0_2_ms|tat_2_4_ms|tat_4_6_ms|tat_6_8_ms|tat_8_10_ms|
# tat_10_12_ms|tat_12_14_ms|tat_14_16_ms|tat_16_18_ms|tat_18_20_ms|tat_20_22_ms|tat_22_24_ms|tat_24_26_ms|tat_26_28_ms|tat_28_30_ms|
# tat_30_35_ms|tat_35_40_ms|tat_40_45_ms|tat_45_50_ms|tat_50_55_ms|tat_55_60_ms|tat_60_65_ms|tat_65_70_ms|tat_70_75_ms|tat_75_80_ms|tat_80_85_ms|
# tat_85_90_ms|tat_90_95_ms|tat_95_100_ms|tat_100_inf_ms
# cache_hit_type: 0 means cache_miss, 1 means cache_hit, 2 means icp_cache_hit
# Let us write a function that gets the parent TAT PDF for cache hits and misses for a given metro

def get_parent_tat_pdf(metro_name: str) -> ProbabilityDensityFunction:
    """Return a parent TAT PDF for ``metro_name`` and ``cache_hit_type``."""
    import pandas as pd

    query = f"""
    SELECT * FROM netopt_perf_midgress_ecor_tat_ansabni
    WHERE edge_metro = '{metro_name}' AND cache_hit_type = 1
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""

    df = None
    conn = sqlite3.connect("PERF/perf_data.db")
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying database: {e}")
        return ProbabilityDensityFunction([])
    finally:
        conn.close()

    if df is None or df.empty:
        print(f"No data found for metro {metro_name} parent TATs ")
        return ProbabilityDensityFunction([])
    
    pdf = ProbabilityDensityFunction.from_dataframe(df)
    return pdf

'''
# Get parent TAT PDFs for each metro and plot them
for metro_name in client_metro_ids:
    for cache_hit_type in [0, 1]:
        parent_tat_pdf = get_parent_tat_pdf(metro_name, cache_hit_type)
        if plt is not None:
            hit_miss_str = "cache_hits" if cache_hit_type == 1 else "cache_misses"
            parent_tat_pdf.plot(title=f"Parent TAT PDF for {metro_name} ({hit_miss_str})")
            plt.savefig(f"tats_parent/parent_tat_pdf_{metro_name}_{hit_miss_str}.png")
            plt.clf()
'''


def get_parent_tat_pdf(metro_name: str) -> ProbabilityDensityFunction:
    """Return a parent TAT PDF for ``metro_name`` and ``cache_hit_type``."""
    import pandas as pd

    query = f"""
    SELECT * FROM netopt_perf_midgress_ecor_tat_ansabni
    WHERE edge_metro = '{metro_name}' AND cache_hit_type != 2
    AND pdate in ('2026-02-07', '2026-02-08', '2026-02-09')"""

    df = None
    conn = sqlite3.connect("PERF/perf_data.db")
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying database: {e}")
        return ProbabilityDensityFunction([])
    finally:
        conn.close()

    if df is None or df.empty:
        return ProbabilityDensityFunction([])
    
    pdf = ProbabilityDensityFunction.from_dataframe(df)
    return pdf

'''
# Get parent TAT PDFs for each metro and plot them
for metro_name in client_metro_ids:
    parent_tat_pdf = get_parent_tat_pdf(metro_name)
    if plt is not None:
        parent_tat_pdf.plot(title=f"Parent TAT PDF for {metro_name}")
        plt.savefig(f"tats_parent_all/parent_tat_pdf_{metro_name}.png")
        plt.clf()
'''

