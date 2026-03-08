#!/usr/bin/env bash
set -euo pipefail

PARTITION_DIR="/home/ubuntu/pringle/sssp/4_BEST"

if [ ! -d "$PARTITION_DIR" ]; then
    echo "Error: partition folder does not exist: $PARTITION_DIR"
    exit 1
fi

PARTITION="custom"
INPUT="/largeTwitchFolder"
OUTPUT_BASE="/outputLargeTwitchFolder"
NODES="/home/ubuntu/pringle/train_test/selected_nodes_64_train0.20_seed0_test_old.txt"
LOG_FILE="$HOME/query_times.log"

shopt -s nullglob
PARTITION_FILES=( "$PARTITION_DIR"/*.txt )

if [ "${#PARTITION_FILES[@]}" -eq 0 ]; then
    echo "Error: no .txt partition files found in $PARTITION_DIR"
    exit 1
fi

for PARTITION_FILE in "${PARTITION_FILES[@]}"; do
    BASENAME="$(basename "$PARTITION_FILE" .txt)"
    OUTPUT="${OUTPUT_BASE}_${BASENAME}"

    echo "=================================================="
    echo "Running partition file: $PARTITION_FILE"
    echo "Output path: $OUTPUT"
    echo "=================================================="

    rm -f /tmp/pringle_query_pipe /tmp/pringle_done

    mpiexec.openmpi -n 16 --oversubscribe --hostfile ~/hosts \
      -x CLASSPATH -x LD_LIBRARY_PATH -x JAVA_HOME \
      ./run setup "$INPUT" "$OUTPUT" "$PARTITION" "$PARTITION_FILE" \
      < /dev/null &
    SETUP_PID=$!

    cleanup() {
        set +e
        ./run teardown >/dev/null 2>&1 || true
        wait "$SETUP_PID" >/dev/null 2>&1 || true
        rm -f /tmp/pringle_query_pipe /tmp/pringle_done
    }
    trap cleanup EXIT

    while [ ! -p /tmp/pringle_query_pipe ]; do
        sleep 1
    done
    echo "Setup ready, running queries..."

    QUERIES_START=$(date +%s%N)
    while read -r src; do
        [ -z "$src" ] && continue
        echo "Running query for source node $src"
        ./run query "$src"
    done < "$NODES"
    QUERIES_END=$(date +%s%N)

    ELAPSED_MS=$(( (QUERIES_END - QUERIES_START) / 1000000 ))
    echo "Total query time for $BASENAME: ${ELAPSED_MS}ms"
    echo "${BASENAME},${PARTITION_FILE},${ELAPSED_MS}" >> "$LOG_FILE"

    ./run teardown
    wait "$SETUP_PID"

    rm -f /tmp/pringle_query_pipe /tmp/pringle_done
    trap - EXIT
done

echo "All partition files completed."
