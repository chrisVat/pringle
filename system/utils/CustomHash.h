#ifndef CUSTOM_HASH_H
#define CUSTOM_HASH_H

#include <unordered_map>
#include <fstream>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>
#include "../utils/global.h"

using json = nlohmann::json;

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
        std::ifstream f("_small_twitch_node_ranks.json");   // <-- worker assignment file path here
        if (!f.is_open()) {
            std::cerr << "Failed to open partition file" << std::endl;
            exit(1);
        }

        json j;
        f >> j;

        for (auto it = j.begin(); it != j.end(); ++it) {
            int node = std::stoi(it.key());
            int rank = it.value();

            if (rank < 0 || rank >= _num_workers) {
                std::cerr << "Invalid worker rank " << rank
                          << " for vertex " << node << std::endl;
                exit(1);
            }

            node_to_rank_[node] = rank;
        }

        std::cout << "Loaded custom partition for "
                  << node_to_rank_.size()
                  << " vertices." << std::endl;
    }
};

template <class KeyT>
std::unordered_map<int, int> CustomHash<KeyT>::node_to_rank_;

#endif
