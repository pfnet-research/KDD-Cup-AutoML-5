#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "aggregate_set.h"

namespace py = pybind11;

std::pair<py::array_t<int>, py::array_t<int>> merge_sorted_index(
  py::array_t<float> src_time_data,
  py::array_t<float> dst_time_data,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index,
  bool is_src_priority
);

std::pair<std::unique_ptr<std::vector<int>>, std::unique_ptr<std::vector<int>>> merge_sorted_index_(
  std::shared_ptr<std::vector<float>> src_time_data,
  std::shared_ptr<std::vector<float>> dst_time_data,
  std::shared_ptr<std::vector<int>> src_sorted_index,
  std::shared_ptr<std::vector<int>> dst_sorted_index,
  bool is_src_priority
);

class Aggregator {
public:
  Aggregator(){};
	std::unique_ptr<std::vector<float>> aggregate_(
    std::unique_ptr<std::vector<float>> dst_data,
    std::unordered_map<int, std::shared_ptr<std::vector<float>>> time_for_each_table,
    std::unordered_map<int, std::shared_ptr<std::vector<int>>> sorted_index_for_each_table,
    std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relation,
    std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relation,
    std::vector<bool> src_is_unique_for_each_relation,
    std::vector<bool> dst_is_unique_for_each_relation,
    std::string mode1,
    std::string mode2);

	py::array_t<float> aggregate(
    py::array_t<float> dst_data,
    std::unordered_map<int, py::array_t<float>> time_for_each_table,
    std::unordered_map<int, py::array_t<int>> sorted_index_for_each_table,
    std::vector<py::array_t<int>> src_id_for_each_relation,
    std::vector<py::array_t<int>> dst_id_for_each_relation,
    std::vector<bool> src_is_unique_for_each_relation,
    std::vector<bool> dst_is_unique_for_each_relation,
    std::string mode1,
    std::string mode2);
};

void InitAggregator(pybind11::module& m);
