#pragma once
#include <string>

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
);
