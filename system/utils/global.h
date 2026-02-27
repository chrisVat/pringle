#ifndef GLOBAL_H
#define GLOBAL_H

#include <unistd.h>
#include <unordered_map>
#include <map>
#include <mpi.h>
#include <stddef.h>
#include <limits.h>
#include <string>
#include <ext/hash_set>
#include <ext/hash_map>
#define hash_map __gnu_cxx::hash_map
#define hash_set __gnu_cxx::hash_set
#include <assert.h> //for ease of debug
using namespace std;

//============================
///worker info
#define MASTER_RANK 0

inline int _my_rank;
inline int _num_workers = 1;

// inline map<pair<int,int>, int> _vertex_comm_map;  // {(src, dst) -> count}

inline unordered_map<int, unordered_map<int, int>> _vertex_comm_map; // unordered_map is not thread safe for multiple workers (unordered_concurrent_map)
// usage: _vertex_comm_map[src_vertex][dst_vertex] += count
// dict[node] -> {paired_node: count}

inline long long _cross_worker_msg_num = 0;
inline vector<vector<int>> _worker_comm_matrix;

inline char _hostname[256];
inline int _machine_id = 0;
inline unordered_map<int, int> _rank_to_machine;  // rank -> machine id

inline long long _cross_machine_msg_num = 0;
inline vector<vector<int>> _machine_comm_matrix;

inline void init_machine_id() 
{
    printf("Rank %d hostname: %s\n", _my_rank, _hostname);
    gethostname(_hostname, sizeof(_hostname)); //gets the current machine's network name that we are running on
    
    // Gather all hostnames to master
    // Master assigns machine IDs based on unique hostnames
    char all_hostnames[_num_workers][256];
    MPI_Allgather(_hostname, 256, MPI_CHAR, all_hostnames, 256, MPI_CHAR, MPI_COMM_WORLD);
    
    // Build hostname -> machine_id mapping
    unordered_map<string, int> host_to_machine;
    int next_machine = 0;
    for (int i = 0; i < _num_workers; i++) {
        string h(all_hostnames[i]); // which machine is worker i on?
        if (host_to_machine.find(h) == host_to_machine.end()) 
            host_to_machine[h] = next_machine++; // first time seeing this hostname, assign new machine id
        _rank_to_machine[i] = host_to_machine[h]; // worker i belongs to this machine
    }
    _machine_id = _rank_to_machine[_my_rank];
    
    if (_my_rank == MASTER_RANK) { // prints machine to worker mapping only in the master
        printf("Machine assignments:\n");
        for (auto& [rank, machine] : _rank_to_machine)
            printf("  Rank %d -> Machine %d (%s)\n", rank, machine, all_hostnames[rank]);
    }
}

inline void init_machine_matrix() 
{
    // Find number of unique machines
    int num_machines = 0;
    for (auto& [rank, machine] : _rank_to_machine) {
        if (machine + 1 > num_machines)
            num_machines = machine + 1;
    }
    
    // Create num_machines x num_machines matrix of zeros
    _machine_comm_matrix.assign(num_machines, vector<int>(num_machines, 0));
    
    if (_my_rank == MASTER_RANK)
        printf("Initialized %dx%d machine comm matrix\n", num_machines, num_machines);
}

inline void init_comm_matrix() 
{
    _worker_comm_matrix.assign(_num_workers, vector<int>(_num_workers, 0));
}

inline int get_worker_id()
{
    return _my_rank;
}
inline int get_num_workers()
{
    return _num_workers;
}

inline void init_workers()
{
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &_num_workers);
    init_machine_id();

    MPI_Comm_rank(MPI_COMM_WORLD, &_my_rank);
    init_comm_matrix();
    init_machine_matrix();
    
    printf("DEBUG: worker %d sees _num_workers=%d\n", _my_rank, _num_workers);
}

inline void worker_finalize()
{
    MPI_Finalize();
}

inline void worker_barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//------------------------
// worker parameters

struct WorkerParams {
    string input_path;
    string output_path;
    bool force_write;
    bool native_dispatcher; //true if input is the output of a previous blogel job

    WorkerParams()
    {
        force_write = true;
        native_dispatcher = false;
    }
};

struct MultiInputParams {
    vector<string> input_paths;
    string output_path;
    bool force_write;
    bool native_dispatcher; //true if input is the output of a previous blogel job

    MultiInputParams()
    {
        force_write = true;
        native_dispatcher = false;
    }

    void add_input_path(string path)
    {
        input_paths.push_back(path);
    }
};

//============================
//general types
typedef int VertexID;

//============================
//global variables
inline int global_step_num;
inline int step_num()
{
    return global_step_num;
}

inline int global_phase_num;
inline int phase_num()
{
    return global_phase_num;
}

inline void* global_message_buffer = NULL;
inline void set_message_buffer(void* mb)
{
    global_message_buffer = mb;
}
inline void* get_message_buffer()
{
    return global_message_buffer;
}

inline void* global_combiner = NULL;
inline void set_combiner(void* cb)
{
    global_combiner = cb;
}
inline void* get_combiner()
{
    return global_combiner;
}

inline void* global_aggregator = NULL;
inline void set_aggregator(void* ag)
{
    global_aggregator = ag;
}
inline void* get_aggregator()
{
    return global_aggregator;
}

inline void* global_agg = NULL; //for aggregator, FinalT of last round
inline void* getAgg()
{
    return global_agg;
}

inline int global_vnum = 0;
inline int& get_vnum()
{
    return global_vnum;
}
inline int global_active_vnum = 0;
inline int& active_vnum()
{
    return global_active_vnum;
}

enum BITS {
    HAS_MSG_ORBIT = 0,
    FORCE_TERMINATE_ORBIT = 1,
    WAKE_ALL_ORBIT = 2
};
//currently, only 3 bits are used, others can be defined by users
inline char global_bor_bitmap;

inline void clearBits()
{
    global_bor_bitmap = 0;
}

inline void setBit(int bit)
{
    global_bor_bitmap |= (2 << bit);
}

inline int getBit(int bit, char bitmap)
{
    return ((bitmap & (2 << bit)) == 0) ? 0 : 1;
}

inline void hasMsg()
{
    setBit(HAS_MSG_ORBIT);
}

inline void wakeAll()
{
    setBit(WAKE_ALL_ORBIT);
}

inline void forceTerminate()
{
    setBit(FORCE_TERMINATE_ORBIT);
}

//====================================================
//Ghost threshold
inline int global_ghost_threshold;

inline void set_ghost_threshold(int tau)
{
    global_ghost_threshold = tau;
}

//====================================================
#define ROUND 11 //for PageRank

#endif
