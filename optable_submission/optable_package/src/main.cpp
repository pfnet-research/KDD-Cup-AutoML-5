#include <pybind11/pybind11.h>
#include <malloc.h>
#include <stdio.h>

#include "aggregator.h"
#include "target_encoder.h"
#include "multi_categorical_manager.h"
#include "factorized_target_encoder.h"
#include "categorical_manager.h"
#include "factorized_manager.h"
#include "sequential_regression.h"
#include "label_encoder_for_multi_fit.h"

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "optable pybind module";
    
    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");
    m.def("malloc_stats", &malloc_stats);
    m.def("malloc_trim", &malloc_trim);

    InitAggregator(m);
    InitTargetEncoder(m);
    InitMultiCategoricalManager(m);
    InitFactorizedTargetEncoder(m);
    InitCategoricalManager(m);
    InitFactorizedManager(m);
    InitSequentialRegression(m);
    InitLabelEncoderForMultiFit(m);
}