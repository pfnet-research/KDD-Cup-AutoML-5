#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // vector用
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include "aggregator.h"
#include "categorical_manager.h"
#include "multi_categorical_manager.h"
#include "redsvd-h/include/RedSVD/RedSVD-h"

namespace py = pybind11;

class FactorizedManager {
public:
  std::vector<CategoricalManager *> categorical_managers;
  std::vector<MultiCategoricalManager *> multi_categorical_managers;
  FactorizedManager(
    std::vector<CategoricalManager *> categorical_managers_,
    std::vector<MultiCategoricalManager *> multi_categorical_managers_
  );
  Eigen::SparseMatrix<float> get_factorized_matrix();
  py::array_t<float> truncated_svd(int rank);
};


void InitFactorizedManager(pybind11::module& m);
