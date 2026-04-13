# I have two tables in SERVEDFROM_DATA/metros_servedfrom_copy.db, demand_overflow_newv2 and provider_metro_forecast
# demand_overflow_newv2 has the following columns:
# asn_provider|asn_provider_name|asn_metro|bw_provider_name|provider|contract|purpose|bw_metro|bucket|quarter|weighted_demand|in_metro_overflow|demand|overflow|bw_metro_level|asn_metro_level|ismanual|mapdate
# I want to find the total demand (it is a column in the above table) per asn_provider, asn_metro, bucket, bw_metro, quarter and write these columns to a new table called demand_summary in the same database. The demand_summary table should have the following columns:
# asn_provider|asn_metro|bucket|bw_metro|quarter|total_demand

'''
import sqlite3

conn = sqlite3.connect("SERVEDFROM_DATA/metros_servedfrom_copy.db")
cursor = conn.cursor()

# Create demand_summary table
cursor.execute("""
CREATE TABLE IF NOT EXISTS demand_summary (
    asn_provider TEXT,
    asn_metro TEXT,
    bucket TEXT,
    bw_metro TEXT,
    quarter TEXT,
    total_demand INTEGER
)
""")

# Insert aggregated data into demand_summary
cursor.execute("""
INSERT INTO demand_summary (asn_provider, asn_metro, bucket, bw_metro, quarter, total_demand)
SELECT asn_provider, asn_metro, bucket, bw_metro, quarter, SUM(demand)
FROM demand_overflow_newv2
GROUP BY asn_provider, asn_metro, bucket, bw_metro, quarter
""")

conn.commit()
conn.close()
'''


# Now, I have another table called provider_metro_forecast with the following columns:
# quarter|country|asn_provider|asn_metro|bucket|traffic
# This gives us the total traffic for the asn_provider in the asn_metro
# This traffic could come from different bw_metros as found in the demand_summary table. The demand column in demand_summary represents the fraction of traffic that is expected to be served from asn_metro for a given bw_metro,
# bucket and quarter. I want to figure out the total traffic served from asn_metro for each bw_metro, bucket and quarter across all asn_providers. 
# I want to write this data to a new table called traffic_summary with the following columns:
# asn_metro|bucket|bw_metro|quarter|total_traffic
# Write code below

import sqlite3

conn = sqlite3.connect("SERVEDFROM_DATA/metros_servedfrom_copy.db")
cursor = conn.cursor()

# Create traffic_summary table
cursor.execute("""
CREATE TABLE IF NOT EXISTS traffic_summary (
    asn_metro TEXT,
    bucket TEXT,
    bw_metro TEXT,
    quarter TEXT,
    total_traffic INTEGER
)
""")

# Insert aggregated data into traffic_summary
cursor.execute("""
INSERT INTO traffic_summary (asn_metro, bucket, bw_metro, quarter, total_traffic)
SELECT ds.asn_metro, ds.bucket, ds.bw_metro, ds.quarter, SUM((ds.total_demand / pmf.traffic) * pmf.traffic)
FROM demand_summary ds
JOIN provider_metro_forecast pmf ON ds.asn_provider = pmf.asn_provider AND ds.asn_metro = pmf.asn_metro AND ds.bucket = pmf.bucket AND ds.quarter = pmf.quarter
GROUP BY ds.asn_metro, ds.bucket, ds.bw_metro, ds.quarter
""")

conn.commit()
conn.close()