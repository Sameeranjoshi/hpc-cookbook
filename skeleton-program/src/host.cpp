#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <map>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>

using ::std::map;
using ::std::optional;
using ::std::string;
using ::std::vector;

using ::poplar::Device;
using ::poplar::DeviceManager;
using ::poplar::Engine;
using ::poplar::FLOAT;
using ::poplar::Graph;
using ::poplar::OptionFlags;
using ::poplar::TargetType;
using ::poplar::Tensor;
using ::poplar::program::Copy;
using ::poplar::program::Program;
using ::poplar::program::Execute;
using ::poplar::program::Repeat;

const auto NUM_DATA_ITEMS = 200000;


struct poplar_matmul_state_t {
    optional<Device> device;
    Graph graph;
    map<string, Tensor> tensors;
    map<string, Program> programs;
    OptionFlags engineOptions;
    map<string, int> programIds;
    vector<Program> programsList;
    vector<float> hostData;
};

auto getIpuDevice(const unsigned int numIpus = 1) -> optional<Device> {
    DeviceManager manager = DeviceManager::createDeviceManager();
    optional<Device> device = std::nullopt;
    for (auto &d : manager.getDevices(TargetType::IPU, numIpus)) {
        std::cout << "Trying to attach to IPU " << d.getId();
        if (d.attach()) {
            std::cout << " - attached" << std::endl;
            device = {std::move(d)};
            break;
        } else {
            std::cout << std::endl << "Error attaching to device" << std::endl;
        }
    }
    return device;
}

auto createGraphAndAddCodelets(poplar_matmul_state_t &state) -> Graph {
    auto graph = poplar::Graph(state.device->getTarget());

    // Add our custom codelet, building from CPP source
    graph.addCodelets({"src/codelets/SkeletonCodelets.cpp"}, "-O3 -I codelets");
    popops::addCodelets(graph);
    return graph;
}

auto buildComputeGraph(poplar_matmul_state_t &state) {
    state.tensors["data"] = state.graph.addVariable(poplar::FLOAT, {NUM_DATA_ITEMS}, "data");
    poputil::mapTensorLinearly(state.graph, state.tensors["data"]);

    const int numTiles = state.device->getTarget().getNumTiles();
    // Add programs and wire up data
    const auto NumElemsPerTile = NUM_DATA_ITEMS / numTiles;
    auto cs = state.graph.addComputeSet("loopBody");
    for (auto tileNum = 0; tileNum < numTiles; tileNum++) {
        const auto sliceEnd = std::min((tileNum + 1) * NumElemsPerTile, (int)NUM_DATA_ITEMS);
        const auto sliceStart = tileNum * NumElemsPerTile;

        auto v = state.graph.addVertex(cs, "SkeletonVertex", {{"data", state.tensors["data"].slice(sliceStart, sliceEnd)}});
        state.graph.setInitialValue(v["howMuchToAdd"], tileNum);
        state.graph.setPerfEstimate(v, 100);
        state.graph.setTileMapping(v, tileNum);
    }
    state.programs["main"] = Repeat(10, Execute(cs));
}

auto defineDataStreams(poplar_matmul_state_t &state) {
    auto toIpuStream = state.graph.addHostToDeviceFIFO("TO_IPU", FLOAT, NUM_DATA_ITEMS);
    auto fromIpuStream = state.graph.addDeviceToHostFIFO("FROM_IPU", FLOAT, NUM_DATA_ITEMS);

    state.programs["copy_to_ipu"] = Copy(toIpuStream, state.tensors["data"]);
    state.programs["copy_to_host"] = Copy(state.tensors["data"], fromIpuStream);
}
