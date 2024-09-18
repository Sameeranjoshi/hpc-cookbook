add_executable(skeleton SkeletonForIpu.cpp)
add_executable(graphcore_dace graphcore_mapping.cpp)
add_executable(graphcore_two_files cpu.cpp)

target_link_libraries(skeleton
        poplar
        poputil
        popops
        )
target_link_libraries(graphcore_dace
        poplar
        poputil
        popops
        )
target_link_libraries(graphcore_two_files
        poplar
        poputil
        popops
        )

configure_file(codelets/SkeletonCodelets.cpp codelets/SkeletonCodelets.cpp COPYONLY)
