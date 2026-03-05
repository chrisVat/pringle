#include "pregel_app_sssp.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define PIPE_PATH "/tmp/pringle_query_pipe"
#define DONE_PATH "/tmp/pringle_done"

// ---------------------------------------------------------------------------
// setup: run with mpiexec. loads graph once, then loops waiting for queries.
// ---------------------------------------------------------------------------
void do_setup(const string& input, const string& output,
              const string& partition, const string& partition_file)
{
    g_use_custom_partition = (partition == "custom");
    g_partition_file = partition_file;

    init_workers();

    if (_my_rank == MASTER_RANK) {
        printf("[setup] Loading graph with %s partitioning...\n",
               g_use_custom_partition ? "custom" : "default");
        fflush(stdout);
    }

    WorkerParams param;
    param.input_path = input;
    param.output_path = output;
    param.force_write = true;
    param.native_dispatcher = false;
    param.source_id = -1;

    SPWorker_pregel worker;
    SPCombiner_pregel combiner;
    worker.setCombiner(&combiner);
    worker.load(param);

    // Create pipe AFTER graph is loaded so source_run.sh only proceeds once truly ready
    if (_my_rank == MASTER_RANK) {
        unlink(PIPE_PATH);
        mkfifo(PIPE_PATH, 0666);
        printf("[setup] Ready for queries.\n");
        fflush(stdout);
    }
    worker_barrier();

    while (true) {
        int src_id = -2;
        if (_my_rank == MASTER_RANK) {
            // blocks until ./run query writes a src_id
            int fd = open(PIPE_PATH, O_RDONLY);
            if (fd < 0) { perror("open pipe"); break; }
            read(fd, &src_id, sizeof(int));
            close(fd);
        }
        MPI_Bcast(&src_id, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

        if (src_id == -1) break;  // teardown sentinel

        if (_my_rank == MASTER_RANK) {
            printf("[setup] Running query src=%d\n", src_id);
            fflush(stdout);
        }

        src = src_id;  // set global read by SPVertex_pregel::compute()

        char out_buf[512];
        snprintf(out_buf, sizeof(out_buf), "%s_%d", output.c_str(), src_id);
        param.output_path = out_buf;
        param.source_id = src_id;

        if (_my_rank == MASTER_RANK) { printf("[debug] calling reset_for_query\n"); fflush(stdout); }
        worker.reset_for_query();
        if (_my_rank == MASTER_RANK) { printf("[debug] calling run_query\n"); fflush(stdout); }
        worker.run_query(param);
        if (_my_rank == MASTER_RANK) { printf("[debug] run_query returned\n"); fflush(stdout); }

        // merge per-worker comm trace files on HDFS
        worker_barrier();
        if (_my_rank == MASTER_RANK) {
            char merge_cmd[512];
            snprintf(merge_cmd, sizeof(merge_cmd),
                "hdfs dfs -getmerge /comm_traces/src_%d/staging/ /tmp/merged_src_%d.csv && "
                "hdfs dfs -put -f /tmp/merged_src_%d.csv /comm_traces/src_%d/merged.csv && "
                "hdfs dfs -rm -r /comm_traces/src_%d/staging/",
                src_id, src_id, src_id, src_id, src_id);
            system(merge_cmd);

            // signal ./run query that this query is done
            FILE* f = fopen(DONE_PATH, "w");
            fclose(f);
        }
    }

    if (_my_rank == MASTER_RANK)
        unlink(PIPE_PATH);

    worker_finalize();
}

// ---------------------------------------------------------------------------
// query: run WITHOUT mpiexec. sends one src_id to the setup process.
// ---------------------------------------------------------------------------
void do_query(int src_id)
{
    int fd = open(PIPE_PATH, O_WRONLY);
    if (fd < 0) {
        perror("Cannot open query pipe (is setup running?)");
        exit(1);
    }
    write(fd, &src_id, sizeof(int));
    close(fd);

    // wait for setup to write the done file
    struct stat st;
    while (stat(DONE_PATH, &st) != 0)
        usleep(100000);  // poll every 100ms
    unlink(DONE_PATH);
    printf("[query] src=%d done\n", src_id);
}

// ---------------------------------------------------------------------------
// teardown: run WITHOUT mpiexec. sends sentinel -1 to shut down setup.
// ---------------------------------------------------------------------------
void do_teardown()
{
    int sentinel = -1;
    int fd = open(PIPE_PATH, O_WRONLY);
    if (fd < 0) {
        perror("Cannot open query pipe (is setup running?)");
        exit(1);
    }
    write(fd, &sentinel, sizeof(int));
    close(fd);
    printf("[teardown] Signal sent.\n");
}

// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        printf("Usage:\n");
        printf("  mpiexec ... ./run setup <input> <output> <partition> [partition_file]\n");
        printf("  ./run query <src_id>\n");
        printf("  ./run teardown\n");
        printf("\n  partition: 'default' (modulo) or 'custom' (file-based)\n");
        return -1;
    }

    string mode = argv[1];

    if (mode == "setup") {
        if (argc < 5) {
            printf("Usage: mpiexec ... ./run setup <input> <output> <partition> [partition_file]\n");
            return -1;
        }
        string input          = argv[2];
        string output         = argv[3];
        string partition      = argv[4];
        string partition_file = (argc >= 6) ? argv[5] : "";
        do_setup(input, output, partition, partition_file);

    } else if (mode == "query") {
        if (argc < 3) {
            printf("Usage: ./run query <src_id>\n");
            return -1;
        }
        do_query(atoi(argv[2]));

    } else if (mode == "teardown") {
        do_teardown();

    } else {
        printf("Unknown mode: %s\n", mode.c_str());
        return -1;
    }

    return 0;
}
