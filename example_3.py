# I have a table in SERVEDFROM_DATA/metros_servedfrom_copy.db called traffic_summary with columns asn_metro|bucket|bw_metro|quarter|total_traffic
# Here asn_metro is the metro where the traffic originates, for bucket always use Disney_Videos, bw_metro is the metro where the traffic is served from, quarter is 1Q26 and total_traffic is the total traffic in Gbps. 
# Wite a function that given a end user metro and a serving metro, returns the total traffic in Gbps for the Disney_Videos bucket for 1Q26. Use sqlite3 to query the database.
from __future__ import annotations

from pathlib import Path
import random

from fds import FootprintDescriptor
from cost import CaribouCostCalculator
from perf_with_mch import MetroPerformanceWithMCH, MCHPerformanceModel
from probability import Convolution, gaussian_pdf, weighted_pdf_sum
from analyse import *
from itertools import product
from collections import defaultdict
import sys

INT32_MAX = 2**31 - 1

_BASE_DIR = Path(__file__).resolve().parent
_FDS_DIR = _BASE_DIR / "FDS"

def get_traffic(asn_metro: str, bw_metro: str) -> float:
	import sqlite3

	db_path = _BASE_DIR / "SERVEDFROM_DATA" / "metros_servedfrom_copy.db"
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	query = """
	SELECT total_traffic
	FROM traffic_summary
	WHERE asn_metro = ? AND bw_metro = ? AND bucket = 'Disney_Videos' AND quarter = '1Q26'
	"""

	cursor.execute(query, (asn_metro, bw_metro))
	result = cursor.fetchone()
	conn.close()

	if result is not None:
		return result[0] * 1000  # total_traffic in Mbps
	else:
		return 0.0  # Return 0 if no data is found for the given metros
	
## Go through the table SERVEDFROM_DATA/metros_servedfrom_copy.db and build a graph that 
# depicts the traffic between metros. Only consider rows where traffic is greater than 2GBPS. Plot the 
# graph where nodes are metros and edges are traffic between them. Use a library like networkx to build the graph and matplotlib to plot it.
# Can you also get the coordinates of the metros using lat long data. The metros are not space seperated by _ separated

#. A file called metro_area.csv is provided in the same directory which has the lat long data for the metros. Use that to plot the graph with the correct coordinates of the metros.
# It has the following columns: "id","metro_area","latitude","longitude","airport_code","country","state","max_distance_from_center"
# Only use metros in the US

'''
Can you add code to build something like this?
Use airport_code as the metro name and latitude and longitude for coordinates. Only consider metros in the US. The graph should have edges between metros where traffic is greater than 2GBPS for Disney_Videos bucket in 1Q26. The weight of the edge should be the total traffic in Gbps.
You should create the following

ALL_METROS = ["LAX", "LAS", "PHX"]

MCH_PARENT_METROS = ["LAX"]

MCH_PARENT_ASSIGNMENT = {
	"LAX": "LAX",
	"LAS": "LAX",
	"PHX": "LAX"
	}

METRO_NAMES = {
	"LAX": "Los_Angeles",
	"LAS": "Las_Vegas",
	"PHX": "Phoenix"
	}

NEIGHBOR_METROS = {
	"LAX": ["LAX", "LAS", "PHX"],
	"LAS": ["LAS", "LAX"],
	"PHX": ["PHX", "LAX"]
	}

'''

def plot_traffic_graph():
    import sqlite3
    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load metro coordinates from CSV
    metro_coords = {}
    metro_df = pd.read_csv(_BASE_DIR / "metro_area.csv")
    for _, row in metro_df.iterrows():
        if row['country'] == 'US':
            metro_coords[row['metro_area']] = (row['latitude'], row['longitude'])

    # Build the traffic graph
    G = nx.DiGraph()
    db_path = _BASE_DIR / "SERVEDFROM_DATA" / "metros_servedfrom_copy.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT asn_metro, bw_metro, total_traffic
    FROM traffic_summary
    WHERE bucket = 'Disney_Videos' AND quarter = '1Q26' AND total_traffic > 5
    """

    cursor.execute(query)
    for asn_metro, bw_metro, total_traffic in cursor.fetchall():
        if asn_metro in metro_coords and bw_metro in metro_coords:
            G.add_edge(asn_metro, bw_metro, weight=total_traffic)

    conn.close()

    # Plot the graph
    pos = {metro: (lon, lat) for metro, (lat, lon) in metro_coords.items()}
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    nx.draw_networkx_edges(G, pos, width=[w/100 for w in weights], alpha=0.5)
    plt.title("Traffic Graph between US Metros for Disney Videos (1Q26)")
    #plt.savefig("traffic_graph.png")
    plt.show()

plot_traffic_graph()