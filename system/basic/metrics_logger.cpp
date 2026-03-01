#include "metrics_logger.h"
#include <fstream>
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

    std::string local_file = "metrics_tmp.csv";
    std::ofstream out(local_file);

    // Write ONLY the data row (no header)
    out << source_id << ","
        << supersteps << ","
        << total_msgs << ","
        << comm_time << ","
        << ser_time << ","
        << trans_time << ","
        << compute_time << ","
        << cross_worker << ","
        << cross_machine << ","
        << ratio << "\n";

    out.close();

    // If global metrics file does not exist, create it with header
    int exists = system("hdfs dfs -test -e /comm_traces/metrics/merged_metrics.csv");

    if (exists != 0) {
        std::ofstream header("metrics_header.csv");
        header << "source,supersteps,total_msgs,"
               << "comm_time,serialization_time,transfer_time,compute_time,"
               << "cross_worker,cross_machine,ratio\n";
        header.close();

        system("hdfs dfs -put metrics_header.csv /comm_traces/metrics/merged_metrics.csv");
        std::remove("metrics_header.csv");
    }

    // Append this run's row
    system(("hdfs dfs -appendToFile " + local_file +
            " /comm_traces/metrics/merged_metrics.csv").c_str());

    std::remove(local_file.c_str());
}
