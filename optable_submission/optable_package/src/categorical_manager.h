#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "aggregator.h"

namespace py = pybind11;

class CategoricalManager {
public:
  CategoricalManager(
    std::vector<std::string> categorical_string_values
  );
  std::unique_ptr<std::vector<int>> label_();
  py::array_t<int> label();
  std::unique_ptr<std::vector<float>> frequency_();
  py::array_t<float> frequency();
  std::unique_ptr<std::vector<int>> is_null_();
  py::array_t<int> is_null();
  std::vector<int> categorical_values;
  std::vector<int> categorical_frequency;
  std::vector<int> sorted_categorical;
  std::vector<std::pair<int, int>> most_common(int n);
  std::unique_ptr<std::vector<int>> is_(int objective_cat_id);
  py::array_t<int> is(int objective_cat_id);
  py::array_t<int> sequential_count_encoding(py::array_t<int> sorted_index, int neighbor);
  std::unique_ptr<std::vector<float>> target_encode_with_dst_induces_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> dst_induces,
    float k
  );
  std::unique_ptr<std::vector<float>> temporal_target_encode_with_dst_induces_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> dst_induces,
    std::shared_ptr<std::vector<float>> time_data,
    std::shared_ptr<std::vector<int>> sorted_index,
    float k
  );
  py::array_t<float> target_encode_with_dst_induces(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    float k
  );
  py::array_t<float> temporal_target_encode_with_dst_induces(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    float k
  );
  std::unique_ptr<std::vector<bool>> calculate_adversarial_valid(
    std::shared_ptr<std::vector<int>> dst_induces,
    std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
    std::shared_ptr<std::vector<int64_t>> adversarial_total_count
  );
  std::unique_ptr<std::vector<float>> target_encode_with_adversarial_regularization_(
    std::unique_ptr<std::vector<float>> targets,
    std::shared_ptr<std::vector<int>> dst_induces,
    std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
    std::shared_ptr<std::vector<int64_t>> adversarial_total_count,
    float k
  );
  std::unique_ptr<std::vector<float>> temporal_target_encode_with_adversarial_regularization_(
    std::unique_ptr<std::vector<float>> targets,
    std::shared_ptr<std::vector<int>> dst_induces,
    std::shared_ptr<std::vector<float>> time_data,
    std::shared_ptr<std::vector<int>> sorted_index,
    std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
    std::shared_ptr<std::vector<int64_t>> adversarial_total_count,
    float k
  );
  py::array_t<float> target_encode_with_adversarial_regularization(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    py::array_t<int64_t> adversarial_true_count,
    py::array_t<int64_t> adversarial_total_count,
    float k
  );
  py::array_t<float> temporal_target_encode_with_adversarial_regularization(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    py::array_t<int64_t> adversarial_true_count,
    py::array_t<int64_t> adversarial_total_count,
    float k
  );
  std::unique_ptr<std::vector<float>> exponential_moving_target_encode_with_dst_induces_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> dst_induces,
    std::shared_ptr<std::vector<float>> time_data,
    std::shared_ptr<std::vector<int>> sorted_index,
    float k,
    float beta
  );
  py::array_t<float> exponential_moving_temporal_target_encode_with_dst_induces(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    float k,
    float beta
  );
  int unique_num = 0;
  int row_num = 0;
  bool has_null = false;
};

void InitCategoricalManager(pybind11::module& m);
