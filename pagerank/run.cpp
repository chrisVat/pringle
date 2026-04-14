#include "pregel_app_pagerank.h"
#include "../system/utils/global.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <cstdio>
#include <cstdlib>

using namespace std;

#define PIPE_PATH "/tmp/pringle_query_pipe"
#define DONE_PATH "/tmp/pringle_done"

// ---------------------------------------------------------------------------
// setup: run with mpiexec. loads graph once, then loops waiting for "run"
// commands from ./run run
// ---------------------------------------------------------------------------
void do_setup(const string& input,
              const string& output,
              const string& partition,
              const string& partition_file,
              bool save_comm_traces)
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
    param.uses_source_id = false;
    param.source_id = -1;
    param.save_comm_traces = save_comm_traces;

    PRWorker_pregel worker;
    PRCombiner_pregel combiner;
    worker.setCombiner(&combiner);

    PRAgg_pregel agg;
    worker.setAggregator(&agg);

    // Load and partition graph once
    worker.load(param);

    if (_my_rank == MASTER_RANK) {
        unlink(PIPE_PATH);
        unlink(DONE_PATH);
        if (mkfifo(PIPE_PATH, 0666) != 0) {
            perror("mkfifo");
            exit(1);
        }
    }

    worker_barrier();

    if (_my_rank == MASTER_RANK) {
        printf("[setup] Ready for PageRank runs.\n");
        fflush(stdout);
    }

    while (true) {
        int cmd = 0;

        if (_my_rank == MASTER_RANK) {
            printf("[setup] Waiting for command on fifo...\n");
            fflush(stdout);

            int fd = open(PIPE_PATH, O_RDONLY);
            if (fd < 0) {
                perror("open pipe");
                break;
            }

            int got = 0;
            char* p = (char*)&cmd;
            while (got < (int)sizeof(int)) {
                int r = read(fd, p + got, sizeof(int) - got);
                if (r < 0) {
                    perror("read pipe");
                    close(fd);
                    exit(1);
                }
                if (r == 0) continue;
                got += r;
            }
            close(fd);
        }

        MPI_Bcast(&cmd, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);

        // -1 means teardown
        if (cmd == -1) {
            break;
        }

        // 1 means run one PageRank trial
        if (cmd != 1) {
            if (_my_rank == MASTER_RANK) {
                printf("[setup] Unknown command %d, ignoring.\n", cmd);
                fflush(stdout);
                FILE* f = fopen(DONE_PATH, "w");
                if (f) fclose(f);
            }
            continue;
        }

        if (_my_rank == MASTER_RANK) {
            printf("[setup] Running one PageRank trial...\n");
            fflush(stdout);
        }

        // Reuse loaded graph, but reset worker/message state
        worker.reset_for_query();

        // run_query() will record timing and reuse the already loaded graph.
        worker.run_query(param);

        // If comm traces are enabled, merge staged PageRank comm trace files
        worker_barrier();
        if (_my_rank == MASTER_RANK && save_comm_traces) {
            char merge_cmd[1024];
            snprintf(
                merge_cmd, sizeof(merge_cmd),
                "hdfs dfs -getmerge /comm_traces/pagerank/staging/ /tmp/merged_pagerank.csv && "
                "hdfs dfs -put -f /tmp/merged_pagerank.csv /comm_traces/pagerank/merged.csv && "
                "hdfs dfs -rm -r /comm_traces/pagerank/staging/"
            );
            system(merge_cmd);
        }

        if (_my_rank == MASTER_RANK) {
            FILE* f = fopen(DONE_PATH, "w");
            if (f) fclose(f);
            printf("[setup] PageRank trial finished.\n");
            fflush(stdout);
        }
    }

    if (_my_rank == MASTER_RANK) {
        unlink(PIPE_PATH);
    }

    worker_finalize();
}

// ---------------------------------------------------------------------------
// run: run WITHOUT mpiexec. sends command=1 to the setup process.
// ---------------------------------------------------------------------------
void do_run()
{
    int cmd = 1;
    int fd = open(PIPE_PATH, O_WRONLY);
    if (fd < 0) {
        perror("Cannot open query pipe (is setup running?)");
        exit(1);
    }

    write(fd, &cmd, sizeof(int));
    close(fd);

    struct stat st;
    while (stat(DONE_PATH, &st) != 0) {
        usleep(100000); // poll every 100ms
    }
    unlink(DONE_PATH);

    printf("[run] PageRank trial done\n");
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
        printf("  mpiexec ... ./run setup <input> <output> <partition> [partition_file] [save_comm_traces]\n");
        printf("  ./run run\n");
        printf("  ./run teardown\n");
        printf("\n  partition: 'default' (modulo) or 'custom' (file-based)\n");
        printf("  save_comm_traces: 0 (false) or 1 (true)\n");
        return -1;
    }

    string mode = argv[1];

    if (mode == "setup") {
        if (argc < 5) {
            printf("Usage: mpiexec ... ./run setup <input> <output> <partition> [partition_file] [save_comm_traces]\n");
            return -1;
        }

        string input          = argv[2];
        string output         = argv[3];
        string partition      = argv[4];
        string partition_file = (argc >= 6) ? argv[5] : "";

        bool save_comm_traces = false;
        if (argc >= 7) {
            save_comm_traces = (atoi(argv[6]) != 0);
        }

        do_setup(input, output, partition, partition_file, save_comm_traces);

    } else if (mode == "run") {
        do_run();

    } else if (mode == "teardown") {
        do_teardown();

    } else {
        printf("Unknown mode: %s\n", mode.c_str());
        return -1;
    }

    return 0;
}