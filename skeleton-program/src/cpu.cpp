#include <iostream>
#include <map>
#include <vector>
#include "host.cpp"  // Include the IPU-related functions

struct poplar_matmul_state_t {
    optional<Device> device;
    Graph graph;
    map<string, Tensor> tensors;
    map<string, Program> programs;
    // Engine engine;  // No need for dynamic allocation here
    OptionFlags engineOptions;
    map<string, int> programIds;
    vector<Program> programsList;
    vector<float> hostData;

};

int main(int argc, char *argv[]) {
    // Dynamic allocation of poplar_matmul_state_t on the heap
    poplar_matmul_state_t* state = new poplar_matmul_state_t();

    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    state->device = getIpuDevice(1);
    if (!state->device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        delete state;  // Free the allocated memory
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    state->graph = createGraphAndAddCodelets(state->device);

    std::cout << "STEP 3: Building the compute graph" << std::endl;
    state->tensors = map<string, Tensor>{};
    state->programs = map<string, Program>{};
    buildComputeGraph(state->graph, state->tensors, state->programs, state->device->getTarget().getNumTiles());

    std::cout << "STEP 4: Define data streams" << std::endl;
    defineDataStreams(state->graph, state->tensors, state->programs);

    // Post step 5 Engine object can't be created on the heap, don't put engine in struct.
    std::cout << "STEP 5: Create engine and compile graph" << std::endl;
    state->engineOptions = OptionFlags{
        {"target.saveArchive", "archive.a"},
        {"debug.instrument", "true"},
        {"debug.instrumentCompute", "true"},
        {"debug.instrumentControlFlow", "true"},
        {"debug.computeInstrumentationLevel", "tile"},
        {"debug.outputAllSymbols", "true"},
        {"autoReport.all", "true"},
        {"autoReport.outputSerializedGraph", "true"},
        {"debug.retainDebugInformation", "true"},
    };

    state->programIds = map<string, int>();
    state->programsList = vector<Program>(state->programs.size());
    int index = 0;
    for (auto &nameToProgram : state->programs) {
        state->programIds[nameToProgram.first] = index;
        state->programsList[index] = nameToProgram.second;
        index++;
    }

    // Now construct the Engine using the constructor
    auto engine = Engine(state->graph, state->programsList, state->engineOptions);

    std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*state->device);
    engine.enableExecutionProfiling();

    std::cout << "STEP 7: Attach data streams" << std::endl;
    state->hostData = vector<float>(NUM_DATA_ITEMS, 1);
    engine.connectStream("TO_IPU", state->hostData.data());
    engine.connectStream("FROM_IPU", state->hostData.data());

    std::cout << "STEP 8: Run programs" << std::endl;
    engine.run(state->programIds["copy_to_ipu"]);  // Copy to IPU
    engine.run(state->programIds["main"]);         // Main program
    engine.run(state->programIds["copy_to_host"]); // Copy from IPU

    // Free the allocated memory
    delete state;

    return EXIT_SUCCESS;
}
