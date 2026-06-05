import csv

target_names = {'San_Antonio', 'Austin', 'Dallas', 'Oklahoma_City', 'Houston'}

with open("SERVEDFROM_DATA/served_from_original.csv", "r", encoding="utf-8") as fin, \
     open("SERVEDFROM_DATA/served_from.csv", "w", newline="", encoding="utf-8") as fout:
    
    reader = csv.reader(fin)
    # The crucial fix: forcing quotes around every single field
    writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
    
    header = next(reader)
    writer.writerow(header)
    
    for row in reader:
        asn_metro = row[0].strip('"')
        bw_metro = row[1].strip('"')
        if asn_metro in target_names and bw_metro in target_names:
            writer.writerow(row)

print("Fixed CSV quotes!")