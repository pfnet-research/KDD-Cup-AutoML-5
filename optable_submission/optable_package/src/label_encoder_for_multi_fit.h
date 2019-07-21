#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "aggregator.h"
#include "redsvd-h/include/RedSVD/RedSVD-h"

namespace py = pybind11;

class LabelEncoderForMultiFit {
private:
  int current_id;
  std::unordered_map<std::string, int> cat_value_to_id;
public:
  LabelEncoderForMultiFit(void);
  void fit(std::vector<std::string>);
  py::array_t<int> transform(std::vector<std::string>);
};

void InitLabelEncoderForMultiFit(pybind11::module& m);
