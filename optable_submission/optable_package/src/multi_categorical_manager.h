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

class MultiCategoricalManager {
public:
  MultiCategoricalManager(
    std::vector<std::string> multi_categorical_string_values
  );
  std::vector<std::unique_ptr<std::unordered_map<int, int>>> multi_categorical_values;
  // std::vector<std::unique_ptr<std::unordered_map<int, float>>> tfidf_values;
  int total_word_num = 0;
  int max_word_num = 0;
  float mean_word_num = 0;
  int unique_word_num = 0;
  int row_num = 0;
  std::vector<float> num_of_words;
  std::vector<float> document_num_of_words;
  std::vector<int> words_sorted_by_num;
  std::vector<int> length_vec;
  // std::vector<float> tfidf_of_1word;
  // std::vector<float> hashed_tfidf;
  int hash_dimension;
  py::array_t<float> target_encode(
    py::array_t<float> targets,
    float k1, float k2
  );
  // py::array_t<float> tfidf_target_encode(
  //   py::array_t<float> targets,
  //   float k1, float k2
  // );
  py::array_t<float> temporal_target_encode(
    py::array_t<float> targets,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    float k1, float k2
  );
  py::array_t<float> target_encode_with_dst_induces(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    float k1, float k2
  );
  py::array_t<float> temporal_target_encode_with_dst_induces(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    float k1, float k2
  );
  std::unique_ptr<std::vector<float>> target_encode_(
    std::unique_ptr<std::vector<float>> targets,
    float k1, float k2
  );
  std::unique_ptr<std::vector<float>> temporal_target_encode_(
    std::unique_ptr<std::vector<float>> targets,
    std::shared_ptr<std::vector<float>> time_data,
    std::shared_ptr<std::vector<int>> sorted_index,
    float k1, float k2
  );
  std::unique_ptr<std::vector<float>> target_encode_with_dst_induces_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> dst_induces,
    float k1, float k2
  );
  std::unique_ptr<std::vector<float>> temporal_target_encode_with_dst_induces_(
    std::unique_ptr<std::vector<float>> targets,
    std::unique_ptr<std::vector<int>> dst_induces,
    std::shared_ptr<std::vector<float>> time_data,
    std::shared_ptr<std::vector<int>> sorted_index,
    float k1, float k2
  );
  // void calculate_hashed_tfidf(
  //   int dimension
  // );
  // py::array_t<float> get_hashed_tfidf(int idx);
  py::array_t<float> length();
  py::array_t<float> nunique();
  py::array_t<float> duplicates();
  // py::array_t<float> tfidf(int idx);
  py::array_t<float> count(int idx);
  py::array_t<float> max_count();
  py::array_t<float> min_count();
  py::array_t<int> mode();
  // py::array_t<int> max_tfidf_words();
  
  py::array_t<float> target_encode_with_adversarial_regularization(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    py::array_t<int64_t> adversarial_true_count,
    py::array_t<int64_t> adversarial_total_count,
    float k1, float k2
  );
  py::array_t<float> temporal_target_encode_with_adversarial_regularization(
    py::array_t<float> targets,
    py::array_t<int> dst_induces,
    py::array_t<float> time_data,
    py::array_t<int> sorted_index,
    py::array_t<int64_t> adversarial_true_count,
    py::array_t<int64_t> adversarial_total_count,
    float k1, float k2
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
    float k1, float k2
  );
  std::unique_ptr<std::vector<float>> temporal_target_encode_with_adversarial_regularization_(
    std::unique_ptr<std::vector<float>> targets,
    std::shared_ptr<std::vector<int>> dst_induces,
    std::shared_ptr<std::vector<float>> time_data,
    std::shared_ptr<std::vector<int>> sorted_index,
    std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
    std::shared_ptr<std::vector<int64_t>> adversarial_total_count,
    float k1, float k2
  );
  std::pair<std::vector<float>, std::vector<float>> get_label_count(
    std::vector<float> labels, std::vector<int> dst_induces
  );
  // py::array_t<float> truncated_svd(int rank, bool tfidf, bool log1p);
  // Eigen::SparseMatrix<float> get_tfidf_matrix();
  Eigen::SparseMatrix<float> get_count_matrix();
};

void InitMultiCategoricalManager(pybind11::module& m);
