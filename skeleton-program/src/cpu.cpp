#include <iostream>
#include <map>
#include <vector>
#include "host.cpp"  // Include the IPU-related functions


// Function to initialize and run the IPU operations
void run_ipu_pipeline_internal(poplar_matmul_state_t* state) {
    // Data initialization
    state->hostData = vector<float>(NUM_DATA_ITEMS, 1);

    // Real code pipeline starts from here.
    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    state->device = getIpuDevice(1);
    if (!state->device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return;
    }
    ////////////////////////////////////////KERNEL
    std::cout << "STEP 2: Building the compute graph" << std::endl;
    kernel_buildComputeGraph(*state);
    ////////////////////////////////////////

    std::cout << "STEP 3: Define data streams" << std::endl;
    defineDataStreams(*state);

    std::cout << "STEP 4: Create engine and compile graph" << std::endl;
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
    state->programsList = vector<Program>(state->programs.size());  // Removing the size causes segfault
    int index = 0;
    for (auto &nameToProgram : state->programs) {
        state->programIds[nameToProgram.first] = index;
        state->programsList[index] = nameToProgram.second;
        index++;
    }

    // Now construct the Engine using the constructor
    auto engine = Engine(state->graph, state->programsList, state->engineOptions);

    std::cout << "STEP 5: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*state->device);
    // engine.enableExecutionProfiling();

    std::cout << "STEP 6: Attach data streams" << std::endl;
    
    engine.connectStream("TO_IPU", state->hostData.data());
    engine.connectStream("FROM_IPU", state->hostData.data());

    std::cout << "STEP 7: Run programs" << std::endl;
    engine.run(state->programIds["copy_to_ipu"]);  // Copy to IPU
    engine.run(state->programIds["main"]);         // Main program
    engine.run(state->programIds["copy_to_host"]); // Copy from IPU
}

int main(int argc, char *argv[]) {
    // Dynamic allocation of poplar_matmul_state_t on the heap
    poplar_matmul_state_t* state = new poplar_matmul_state_t();

    run_ipu_pipeline_internal(state);  // Call the function to run the IPU pipeline

    // Free the allocated memory
    delete state;

    return EXIT_SUCCESS;
}
