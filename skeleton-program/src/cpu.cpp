#include <iostream>
#include <map>
#include <vector>
#include "host.cpp"  // Include the IPU-related functions


int main(int argc, char *argv[]) {
    std::cout << "STEP 1: Connecting to an IPU device" << std::endl;
    auto device = getIpuDevice(1);
    if (!device.has_value()) {
        std::cerr << "Could not attach to an IPU device. Aborting" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "STEP 2: Create graph and compile codelets" << std::endl;
    auto graph = createGraphAndAddCodelets(device);

    std::cout << "STEP 3: Building the compute graph" << std::endl;
    auto tensors = map<string, Tensor>{};
    auto programs = map<string, Program>{};
    buildComputeGraph(graph, tensors, programs, device->getTarget().getNumTiles());

    std::cout << "STEP 4: Define data streams" << std::endl;
    defineDataStreams(graph, tensors, programs);

    std::cout << "STEP 5: Create engine and compile graph" << std::endl;
    auto ENGINE_OPTIONS = OptionFlags{
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

    auto programIds = map<string, int>();
    auto programsList = vector<Program>(programs.size());
    int index = 0;
    for (auto &nameToProgram : programs) {
        programIds[nameToProgram.first] = index;
        programsList[index] = nameToProgram.second;
        index++;
    }
    auto engine = Engine(graph, programsList, ENGINE_OPTIONS);

    std::cout << "STEP 6: Load compiled graph onto the IPU tiles" << std::endl;
    engine.load(*device);
    engine.enableExecutionProfiling();

    std::cout << "STEP 7: Attach data streams" << std::endl;
    auto hostData = vector<float>(NUM_DATA_ITEMS, 1);
    engine.connectStream("TO_IPU", hostData.data());
    engine.connectStream("FROM_IPU", hostData.data());

    std::cout << "STEP 8: Run programs" << std::endl;
    engine.run(programIds["copy_to_ipu"]);  // Copy to IPU
    engine.run(programIds["main"]);         // Main program
    engine.run(programIds["copy_to_host"]); // Copy from IPU

    return EXIT_SUCCESS;
}
