#!/bin/bash
set -euo pipefail

PARTITION="custom"   # "default" or "custom"
PARTITION_FILE="/home/ubuntu/pringle/pagerank/topology_partition_15m_4w.txt"
INPUT="/largeTwitchFolder"
OUTPUT="/outputLargeTwitchFolder"
SAVE_COMM_TRACES=0
RUNS=50

rm -f /tmp/pringle_query_pipe /tmp/pringle_done /tmp/query_times_*.csv

if [ "$PARTITION" = "custom" ]; then
  mpiexec.openmpi -n 60 --oversubscribe --hostfile ~/hosts \
    -x CLASSPATH -x LD_LIBRARY_PATH -x JAVA_HOME \
    ./run setup "$INPUT" "$OUTPUT" "$PARTITION" "$PARTITION_FILE" "$SAVE_COMM_TRACES" \
    < /dev/null &
else
  mpiexec.openmpi -n 60 --oversubscribe --hostfile ~/hosts \
    -x CLASSPATH -x LD_LIBRARY_PATH -x JAVA_HOME \
    ./run setup "$INPUT" "$OUTPUT" "$PARTITION" "" "$SAVE_COMM_TRACES" \
    < /dev/null &
fi

SETUP_PID=$!

while [ ! -p /tmp/pringle_query_pipe ]; do sleep 1; done
echo "Setup ready, running PageRank $RUNS times..."

RUNS_START=$(date +%s%N)

for i in $(seq 1 $RUNS); do
  echo "Running PageRank trial $i / $RUNS"
  ./run run
done

RUNS_END=$(date +%s%N)
ELAPSED_MS=$(( (RUNS_END - RUNS_START) / 1000000 ))

echo "Total PageRank trial time: ${ELAPSED_MS}ms"
echo "Total PageRank trial time: ${ELAPSED_MS}ms" >> ~/query_times.log

./run teardown
wait $SETUP_PID