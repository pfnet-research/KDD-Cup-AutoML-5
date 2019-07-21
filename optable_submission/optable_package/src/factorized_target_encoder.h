#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "aggregator.h"

namespace py = pybind11;


class FactorizedTargetEncoder {
public:
  FactorizedTargetEncoder(){};
  py::array_t<float> encode(
    py::array_t<float> targets,
    py::array_t<int> ids1,
    py::array_t<int> ids2,
    float k0, float k1, float k2
  );
  py::array_t<float> temporal_encode(
    py::array_t<float> targets,
    py::array_t<int> ids1,
    py::array_t<int> ids2,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    float k0, float k1, float k2
  );
};

void InitFactorizedTargetEncoder(pybind11::module& m);
