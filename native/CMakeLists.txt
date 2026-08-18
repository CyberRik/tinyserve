# TinyServe native prefix cache.
#
# Builds two things from one source file:
#   tinyserve_prefix  -- shared library, loaded by tinyserve/prefix/cache.py via ctypes
#   ts_prefix_tests   -- standalone C++ correctness harness, no Python involved
#
# There is no Python.h and no pybind11 here on purpose. A flat C ABI loaded over
# ctypes keeps the build independent of which interpreter (and which ABI tag) is
# on the host, and leaves the same translation unit linkable straight into a C++
# server. See include/tinyserve/prefix_cache.h for the full rationale.

cmake_minimum_required(VERSION 3.15)
project(tinyserve_prefix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -fvisibility=hidden pairs with TINYSERVE_API in the header: only the ts_* entry
# points land in the dynamic symbol table, so the ctypes surface is exactly the
# documented ABI and nothing else.
if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wpedantic -fvisibility=hidden)
else()
  add_compile_options(/W4)
endif()

add_library(tinyserve_prefix SHARED src/prefix_cache.cpp)
target_include_directories(tinyserve_prefix PUBLIC include)
target_compile_definitions(tinyserve_prefix PRIVATE TINYSERVE_BUILD_SHARED)
set_target_properties(tinyserve_prefix PROPERTIES
  # cache.py probes a fixed set of names; keep the bare name on every platform
  # so the loader does not need per-platform prefix rules.
  PREFIX ""
  POSITION_INDEPENDENT_CODE ON
)

# Static, so the harness tests the same code the shared library ships without
# needing the .so on the loader path.
add_executable(ts_prefix_tests tests/test_prefix_cache.cpp src/prefix_cache.cpp)
target_include_directories(ts_prefix_tests PRIVATE include)

enable_testing()
add_test(NAME prefix_cache COMMAND ts_prefix_tests)
