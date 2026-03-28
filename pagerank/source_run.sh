#!/bin/bash

PARTITION="default"   # change to "custom" to use file-based partitioning
PARTITION_FILE="/home/ubuntu/pringle/pagerank/theboogalo3__1_3_1000.txt"
INPUT="/largeTwitchFolder"
OUTPUT="/outputLargeTwitchFolder"

START=$(date +%s%N)

if [ "$PARTITION" = "custom" ]; then
  mpiexec.openmpi -n 60 --oversubscribe --hostfile ~/hosts \
    -x CLASSPATH -x LD_LIBRARY_PATH -x JAVA_HOME \
    ./run "$INPUT" "$OUTPUT" "$PARTITION" "$PARTITION_FILE"
else
  mpiexec.openmpi -n 60 --oversubscribe --hostfile ~/hosts \
    -x CLASSPATH -x LD_LIBRARY_PATH -x JAVA_HOME \
    ./run "$INPUT" "$OUTPUT" "$PARTITION"
fi

END=$(date +%s%N)

ELAPSED_MS=$(( (END - START) / 1000000 ))
echo "Total PageRank run time: ${ELAPSED_MS}ms"
echo "Total PageRank run time: ${ELAPSED_MS}ms" >> ~/query_times.log