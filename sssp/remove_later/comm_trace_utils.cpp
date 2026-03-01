#include "comm_trace_utils.h"
#include <fstream>
#include <cstdio>
#include <string>

using namespace std;

void merge_worker_files(int source, int num_workers) {
    bool write_header = false;

    // Check if file already exists
    ifstream check("vertex_comm_all.csv");
    if(!check.good()) {
        write_header = true;
    }
    check.close();

    ofstream global("vertex_comm_all.csv", ios::app);

    for(int rank = 0; rank < num_workers; rank++) {
        char filename[128];
        sprintf(filename, "vertex_comm_worker_%d_src_%d.csv", rank, source);

        ifstream in(filename);
        if(!in.is_open()) continue;

        string line;

        // Read header from worker
        if(getline(in, line)) {
            if(write_header) {
                global << line << "\n";
                write_header = false;
            }
        }

        while(getline(in, line)) {
            global << line << "\n";
        }

        in.close();
        remove(filename);
    }

    global.close();
}
