#include <iostream>
#include <random>
#include <string>

#include "HyGraph/hybrid_block_graph.hpp"
#include "HyGraph/program.hpp"

using namespace std;
using namespace hygraph;

class MyProgram: public Program<MyProgram, int, float, float, float> {
    public:
    static const ActivityType activity = ACTIVITY_SELECTED;

    int root;

    MyProgram(int root_id) {
        root = root_id;
    }

    /**
     * Set the initial state of each vertex. Return true if the vertex needs to be activated.
     */
    INLINE CUDA_HOST_DEVICE bool init_state(const int id, float &state) const {
        if (id == root) {
            state = 0.0;
            return true;
        } else {
            state = FLT_MAX;
            return false;
        }
    }

    /**
     * Initialize an empty message. Used as the initial value when aggregating messages.
     */
    INLINE CUDA_HOST_DEVICE void init_message(float &msg) const {
        msg = FLT_MAX;
    }

    /**
     * Generate the message of a vertex that will be send to all neighbors.
     */
    INLINE CUDA_HOST_DEVICE bool generate_message(int superstep, const int id, const float state, float &msg) const {
        msg = state;
        return true;
    }

    /**
     * Modify contents of message based on edge value.
     */
    INLINE CUDA_HOST_DEVICE bool process_edge(int superstep, const float &edge_weight, float &msg) const {
        msg += edge_weight;
        return true;
    }

    /**
     * Combine two incoming messages for a vertex into one message.
     */
    INLINE CUDA_HOST_DEVICE void aggregate(const float &msg, float &result) const {
        if (msg < result) {
            result = msg;
        }
    }

    /**
     * Process the incoming message for a vertex. Return true if the vertex needs to be activated, false otherwise.
     */
    INLINE CUDA_HOST_DEVICE bool process_vertex(int superstep, const int id, const float msg, float &state) const {
        if (msg < state) {
            state = msg;
            return true;
        } else {
            return false;
        }
    }

};

void generate_graph(size_t scale, hygraph::LocalGraph<int, float> &graph) {
    size_t n = 1 << scale;
    size_t m = n * 10;

    log("generating graph: %lld vertices, %lld edges",
            (long long int) n, (long long int) m);

    std::random_device dev;
    std::mt19937 rd(dev());
    std::uniform_int_distribution<vid_t> dist(0, (vid_t) n - 1);

    for (size_t i = 0; i < n; i++) {
        graph.add_vertex(i);
    }

    // add remaining vertices.
    for (size_t i = 0; i < m; i++) {
        graph.add_edge(dist(rd), dist(rd), dist(rd));
    }

    // add edges between vertices in a ring to ensure they are all connected.
    for (size_t i = 0; i < n; i++) {
        graph.add_edge(i, (i + 1) % n, dist(rd));
    }

    graph.remove_duplicate_edges();
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "usage: " << argv[0] << " <num-vertices>" << endl;
        return EXIT_FAILURE;
    }

    size_t scale = strtoll(argv[1], NULL, 10);

    if (scale == 0) {
        cerr << "error: scale should be > 0" << endl;
        return EXIT_FAILURE;
    }

    if (scale >= 31) {
        cerr << "error: too many vertices" << endl;
        return EXIT_FAILURE;
    }

    hygraph::LocalGraph<int, float> local_graph;
    generate_graph(scale, local_graph);

    hygraph::HybridBlockGraph<MyProgram> hybrid_graph(0, 32);
    hybrid_graph.load(local_graph, 1024);

    MyProgram program(0);

    log("==================================");
    log(" CPU-only ");
    log("==================================");
    hybrid_graph.run_hybrid_static(program, 50, 1.0);
    log("");

    log("==================================");
    log(" GPU-only ");
    log("==================================");
    hybrid_graph.run_hybrid_static(program, 50, 0.0);
    log("");

    log("==================================");
    log(" CPU+GPU ");
    log("==================================");
    hybrid_graph.run_hybrid_dynamic(program, 50);
    log("");
}
