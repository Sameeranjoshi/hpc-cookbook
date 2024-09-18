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

auto createGraphAndAddCodelets(const optional<Device> &device) -> Graph {
    auto graph = poplar::Graph(device->getTarget());

    // Add our custom codelet, building from CPP source
    graph.addCodelets({"src/codelets/SkeletonCodelets.cpp"}, "-O3 -I codelets");
    popops::addCodelets(graph);
    return graph;
}

auto buildComputeGraph(Graph &graph, map<string, Tensor> &tensors,
                       map<string, Program> &programs, const int numTiles) {
    tensors["data"] = graph.addVariable(poplar::FLOAT, {NUM_DATA_ITEMS}, "data");
    poputil::mapTensorLinearly(graph, tensors["data"]);

    // Add programs and wire up data
    const auto NumElemsPerTile = NUM_DATA_ITEMS / numTiles;
    auto cs = graph.addComputeSet("loopBody");
    for (auto tileNum = 0; tileNum < numTiles; tileNum++) {
        const auto sliceEnd = std::min((tileNum + 1) * NumElemsPerTile, (int)NUM_DATA_ITEMS);
        const auto sliceStart = tileNum * NumElemsPerTile;

        auto v = graph.addVertex(cs, "SkeletonVertex", {{"data", tensors["data"].slice(sliceStart, sliceEnd)}});
        graph.setInitialValue(v["howMuchToAdd"], tileNum);
        graph.setPerfEstimate(v, 100);
        graph.setTileMapping(v, tileNum);
    }
    programs["main"] = Repeat(10, Execute(cs));
}

auto defineDataStreams(Graph &graph, map<string, Tensor> &tensors,
                       map<string, Program> &programs) {
    auto toIpuStream = graph.addHostToDeviceFIFO("TO_IPU", FLOAT, NUM_DATA_ITEMS);
    auto fromIpuStream = graph.addDeviceToHostFIFO("FROM_IPU", FLOAT, NUM_DATA_ITEMS);

    programs["copy_to_ipu"] = Copy(toIpuStream, tensors["data"]);
    programs["copy_to_host"] = Copy(tensors["data"], fromIpuStream);
}
