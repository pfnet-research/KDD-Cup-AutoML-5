#include "label_encoder_for_multi_fit.h"


LabelEncoderForMultiFit::LabelEncoderForMultiFit(void){
    this->current_id = 0;
    this->cat_value_to_id = std::unordered_map<std::string, int>{};
}

void LabelEncoderForMultiFit::fit(std::vector<std::string> cat_values){
    int row_num = cat_values.size();
    for(int row_idx = 0; row_idx < row_num; row_idx++) {
        std::string* cat_value = (cat_values.data() + row_idx);
        if((*cat_value) != ""){
            if (this->cat_value_to_id.find((*cat_value)) == this->cat_value_to_id.end()){
                this->cat_value_to_id[(*cat_value)] = this->current_id;
                this->current_id++;
            }
        }
    }
}

py::array_t<int> LabelEncoderForMultiFit::transform(std::vector<std::string> cat_values){
    std::vector<int> ret_vec(cat_values.size());
    int row_num = cat_values.size();
    for(int row_idx = 0; row_idx < row_num; row_idx++) {
        std::string* cat_value = (cat_values.data() + row_idx);
        if((*cat_value) != ""){
            if (this->cat_value_to_id.find((*cat_value)) == this->cat_value_to_id.end()){
                ret_vec[row_idx] = -1;
            } else {
                ret_vec[row_idx] = this->cat_value_to_id[(*cat_value)];
            }
        } else {
            ret_vec[row_idx] = -1;
        }
    }
    py::array_t<int> ret = py::array_t<int>(ret_vec.size(), ret_vec.data());
    return ret;
}

void InitLabelEncoderForMultiFit(pybind11::module& m){
  m.doc() = "label encoder for multi fit made by pybind11"; // optional

  py::class_<LabelEncoderForMultiFit>(m, "LabelEncoderForMultiFit")
    .def(py::init<>())
    .def("fit", &LabelEncoderForMultiFit::fit)
    .def("transform", &LabelEncoderForMultiFit::transform);
}
