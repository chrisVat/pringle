while read src; do
  mpiexec.openmpi -n 16 --oversubscribe --hostfile ~/hosts \
    -x CLASSPATH -x LD_LIBRARY_PATH -x JAVA_HOME \
    ./run $src /largeTwitchFolder /outputLargeTwitchFolder_$src
done < /home/ubuntu/pringle/train_test/selected_nodes_64_train0.20_seed0_train.txt
