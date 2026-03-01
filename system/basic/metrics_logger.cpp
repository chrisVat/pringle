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

    bool exists = file_exists("sssp_metrics.csv");
    std::ofstream metrics("sssp_metrics.csv", std::ios::app);

    if (!exists) {
        metrics << "source,supersteps,total_msgs,"
                << "comm_time,serialization_time,transfer_time,compute_time,"
                << "cross_worker,cross_machine,ratio\n";
    }

    metrics << source_id << ","
            << supersteps << ","
            << total_msgs << ","
            << comm_time << ","
            << ser_time << ","
            << trans_time << ","
            << compute_time << ","
            << cross_worker << ","
            << cross_machine << ","
            << ratio << "\n";

    metrics.close();
}
