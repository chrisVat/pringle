#!/bin/bash
set -euo pipefail

PARTITION="custom"   # or "custom"
PARTITION_FILE="/home/ubuntu/pringle/pagerank/compute_only_pagerank_15m_4w.txt"
INPUT="/largeTwitchFolder"
OUTPUT="/outputLargeTwitchFolder"
RUNS=50

# Clear old timing file so this experiment is clean
rm -f /tmp/pagerank_times.csv

for i in $(seq 1 $RUNS); do
  echo "===== PageRank run $i / $RUNS ====="

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
  echo "Wrapper wall time run $i: ${ELAPSED_MS}ms"
done

echo
echo "Finished $RUNS PageRank runs."
echo "Worker.h should have appended per-run data to /tmp/pagerank_times.csv"