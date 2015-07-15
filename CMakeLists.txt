SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})


CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)

IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_subdirectory (cnpy)

FIND_PACKAGE(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIR} "cnpy/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

LINK_DIRECTORIES(${TORCH_LIBRARY_DIR})


SET(src npy4th.cpp)
SET(luasrc init.lua)
ADD_TORCH_PACKAGE(npy4th "${src}" "${luasrc}" "Compatibility Tools")
target_link_libraries(npy4th cnpy TH luaT z)
