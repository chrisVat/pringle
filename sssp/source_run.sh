PARTITION="custom"   # change to "custom" to use file-based partitioning
PARTITION_FILE="/home/ubuntu/pringle/sssp/theboogalo3__1_3_1000.txt"
INPUT="/largeTwitchFolder"
OUTPUT="/outputLargeTwitchFolder"
NODES="/home/ubuntu/pringle/train_test/selected_nodes_64_train0.20_seed0_test.txt"

# Clean up any stale state from a previous run
rm -f /tmp/pringle_query_pipe /tmp/pringle_done

# Launch setup in background: loads + partitions graph once, then waits
mpiexec.openmpi -n 16 --oversubscribe --hostfile ~/hosts \
  -x CLASSPATH -x LD_LIBRARY_PATH -x JAVA_HOME \
  ./run setup $INPUT $OUTPUT $PARTITION $PARTITION_FILE \
  < /dev/null &
SETUP_PID=$!

# Wait until setup signals it is ready (pipe file appears)
while [ ! -p /tmp/pringle_query_pipe ]; do sleep 1; done
echo "Setup ready, running queries..."

# Time just the queries (no setup/teardown cost)
QUERIES_START=$(date +%s%N)
while read src; do
  echo "Running query for source node $src"
  ./run query $src
done < $NODES
QUERIES_END=$(date +%s%N)

ELAPSED_MS=$(( (QUERIES_END - QUERIES_START) / 1000000 ))
echo "Total query time: ${ELAPSED_MS}ms"
echo "Total query time: ${ELAPSED_MS}ms" >> ~/query_times.log

# Shut down the persistent workers
./run teardown
wait $SETUP_PID
