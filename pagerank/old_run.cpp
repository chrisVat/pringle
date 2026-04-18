#include "pregel_app_pagerank.h"
#include "../system/utils/global.h"
#include <string>
#include <cstdio>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: mpiexec ... ./run <input> <output> <partition> [partition_file]\n");
        printf("  partition: 'default' (modulo) or 'custom' (file-based)\n");
        return -1;
    }

    string input          = argv[1];
    string output         = argv[2];
    string partition      = argv[3];
    string partition_file = (argc >= 5) ? argv[4] : "";

    g_use_custom_partition = (partition == "custom");
    g_partition_file = partition_file;

    init_workers();

    if (_my_rank == MASTER_RANK) {
        printf("[pagerank] Loading graph with %s partitioning...\n",
               g_use_custom_partition ? "custom" : "default");
        fflush(stdout);
    }

    pregel_pagerank(input, output, true, false);

    worker_finalize();
    return 0;
}