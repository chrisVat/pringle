#include "metrics_logger.h"
#include <sys/stat.h>

bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

void write_metrics(
    int source_id,
    long long supersteps,
    long long total_msgs,
    double comm_time,
    double ser_time,
    double trans_time,
    double compute_time,
    long long cross_worker,
    long long cross_machine
) {
    double ratio = 0.0;
    if (cross_worker > 0)
        ratio = (double)cross_machine / cross_worker;

    // Ensure metrics directory exists
    system("hdfs dfs -mkdir -p /comm_traces/metrics/");

    const char* local_file = "metrics_tmp.csv";
    FILE* out = fopen(local_file, "w"); 
    fprintf(out, "%d,%lld,%lld,%f,%f,%f,%f,%lld,%lld,%f\n",
            source_id, supersteps, total_msgs,
            comm_time, ser_time, trans_time, compute_time,
            cross_worker, cross_machine, ratio);
    fclose(out);

    // If global metrics file does not exist, create it with header
    int exists = system("hdfs dfs -test -e /comm_traces/metrics/merged_metrics.csv");

    if (exists != 0) {
        FILE* header = fopen("metrics_header.csv", "w");
        fprintf(header, "source,supersteps,total_msgs,"
                        "comm_time,serialization_time,transfer_time,compute_time,"
                        "cross_worker,cross_machine,ratio\n");
        fclose(header);

        system("hdfs dfs -put metrics_header.csv /comm_traces/metrics/merged_metrics.csv");
        remove("metrics_header.csv");
    }

    // Append this run's row
    char append_cmd[512];
    sprintf(append_cmd, "hdfs dfs -appendToFile %s /comm_traces/metrics/merged_metrics.csv", local_file);
    system(append_cmd);
    remove(local_file);
}
