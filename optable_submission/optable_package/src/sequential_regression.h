#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "aggregator.h"

namespace py = pybind11;

typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::mean>
  > MEAN_ACC_SET;


py::array_t<float> sequential_regression_aggregation(
  py::array_t<float> dst_data,
  py::array_t<int> src_id,
  py::array_t<int> dst_id,
  py::array_t<float> src_time,
  py::array_t<float> dst_time,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index
);

void InitSequentialRegression(pybind11::module& m);