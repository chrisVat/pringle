#!/bin/bash
TMP="/tmp/all_merged.csv"

# Write header once
hdfs dfs -cat /comm_traces/src_47211/merged.csv | head -1 > $TMP

# Append all files skipping their headers
for src_dir in $(hdfs dfs -ls /comm_traces | grep src_ | awk '{print $8}'); do
    echo "Adding $src_dir..."
    hdfs dfs -cat $src_dir/merged.csv | tail -n +2 >> $TMP
done

echo "Total lines: $(wc -l < $TMP)"

# Push combined file to HDFS
hdfs dfs -put -f $TMP /comm_traces/all_merged.csv
echo "Done: $(hdfs dfs -ls /comm_traces/all_merged.csv)"