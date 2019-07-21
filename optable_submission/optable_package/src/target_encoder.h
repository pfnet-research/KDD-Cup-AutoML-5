#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "aggregator.h"

namespace py = pybind11;

class TargetEncoder {
public:
  TargetEncoder(){};
  std::unique_ptr<std::vector<float>> encode_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> ids,
    int k
  );
	py::array_t<float> encode(
    py::array_t<float> targets,
    py::array_t<int> ids,
    int k
  );
  std::unique_ptr<std::vector<float>> not_loo_encode_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> ids,
    int k
  );
	py::array_t<float> not_loo_encode(
    py::array_t<float> targets,
    py::array_t<int> ids,
    int k
  );
  std::unique_ptr<std::vector<float>> temporal_encode_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> ids,
    std::shared_ptr<std::vector<float>> time_data,
    std::shared_ptr<std::vector<int>> sorted_index,
    int k
  );
	py::array_t<float> temporal_encode(
    py::array_t<float> targets,
    py::array_t<int> ids,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    int k
  );
};

void InitTargetEncoder(pybind11::module& m);
