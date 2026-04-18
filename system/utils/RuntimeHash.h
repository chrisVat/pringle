#ifndef RUNTIME_HASH_H
#define RUNTIME_HASH_H

#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "../utils/global.h"

// Single hash class that selects between modulo (default) and file-based
// (custom) partitioning at runtime based on g_use_custom_partition and
// g_partition_file. Set those globals before constructing the Worker.
template <class KeyT>
class RuntimeHash {
public:
    RuntimeHash() {
        if (g_use_custom_partition && !loaded_) {
            load_mapping(g_partition_file);
            loaded_ = true;
        }
    }

    inline int operator()(KeyT key) {
        if (g_use_custom_partition) {
            auto it = node_to_rank_.find(key);
            if (it != node_to_rank_.end()) return it->second;
            std::cerr << "Missing partition entry for vertex " << key << std::endl;
            exit(1);
        } else {
            return (key >= 0) ? key % _num_workers : (-key) % _num_workers;
        }
    }

private:
    static std::unordered_map<int, int> node_to_rank_;
    static bool loaded_;

    void load_mapping(const std::string& path) {
        if (path.empty()) {
            std::cerr << "Custom partition selected but no partition file specified." << std::endl;
            exit(1);
        }
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Failed to open partition file: " << path << std::endl;
            exit(1);
        }
        int node, rank;
        size_t count = 0;
        while (f >> node >> rank) {
            if (rank < 0 || rank >= _num_workers) {
                std::cerr << "Invalid rank " << rank << " for vertex " << node << std::endl;
                exit(1);
            }
            node_to_rank_[node] = rank;
            count++;
        }
        std::cout << "Loaded custom partition for " << count
                  << " vertices from " << path << std::endl;
    }
};

template <class KeyT> std::unordered_map<int, int> RuntimeHash<KeyT>::node_to_rank_;
template <class KeyT> bool RuntimeHash<KeyT>::loaded_ = false;

#endif
