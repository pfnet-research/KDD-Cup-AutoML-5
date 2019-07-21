cmake_minimum_required(VERSION 2.8.12)
project(optable)

add_subdirectory(pybind11)
pybind11_add_module(_core MODULE
  main.cpp
  aggregate_set.cpp
  aggregator.cpp
  target_encoder.cpp
  multi_categorical_manager.cpp
  factorized_target_encoder.cpp
  categorical_manager.cpp
  factorized_manager.cpp
  sequential_regression.cpp
  label_encoder_for_multi_fit.cpp
  )

FIND_PACKAGE(Boost)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()

INCLUDE_DIRECTORIES(/usr/include/eigen3)