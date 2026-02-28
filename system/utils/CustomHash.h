#ifndef CUSTOM_HASH_H
#define CUSTOM_HASH_H

#include <unordered_map>
#include <fstream>
#include <string>
#include <iostream>
#include <cstdlib>
#include "../utils/global.h"

template <class KeyT>
class CustomHash {
public:
    CustomHash() {
        static bool loaded = false;
        if (!loaded) {
            load_mapping();
            loaded = true;
        }
    }

    inline int operator()(KeyT key)
    {
        auto it = node_to_rank_.find(key);
        if (it != node_to_rank_.end()) {
            return it->second;
        }

        std::cerr << "Missing partition entry for vertex "
                  << key << std::endl;
        exit(1);
    }

private:
    static std::unordered_map<int, int> node_to_rank_;

    void load_mapping()
    {
        std::ifstream f("_small_twitch_node_ranks.txt");   // <-- worker assignment file path here
        if (!f.is_open()) {
            std::cerr << "Failed to open partition file" << std::endl;
            exit(1);
        }

        int node, rank;
        size_t count = 0;

        while (f >> node >> rank) {

            if (rank < 0 || rank >= _num_workers) {
                std::cerr << "Invalid worker rank "
                          << rank << " for vertex "
                          << node << std::endl;
                std::exit(1);
            }

            node_to_rank_[node] = rank;
            count++;
        }

        std::cout << "Loaded custom partition for "
                  << count << " vertices." << std::endl;
    }
};

template <class KeyT>
std::unordered_map<int, int> CustomHash<KeyT>::node_to_rank_;

#endif
