#pragma once
#include <string>
#include "utils/global.h"

void write_metrics(
    const WorkerParams& params,
    long long supersteps,
    long long total_msgs,
    double comm_time,
    double ser_time,
    double trans_time,
    double compute_time,
    long long cross_worker,
    long long cross_machine
);
