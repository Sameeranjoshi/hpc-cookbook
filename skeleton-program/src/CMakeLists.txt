add_executable(skeleton SkeletonForIpu.cpp)
add_executable(copy_a_b copy_a_b_skeletonIPU.cpp)

target_link_libraries(skeleton
        poplar
        poputil
        popops
        )
target_link_libraries(copy_a_b
	poplar
	poputil
	popops
	)
configure_file(codelets/SkeletonCodelets.cpp codelets/SkeletonCodelets.cpp COPYONLY)
