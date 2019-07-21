#include "factorized_target_encoder.h"


py::array_t<float> FactorizedTargetEncoder::encode(
  py::array_t<float> targets,
  py::array_t<int> ids1,
  py::array_t<int> ids2,
  float k0, float k1, float k2
){
  const int size = targets.shape(0);
  auto targets_r = targets.unchecked<1>();

  py::array_t<float> ret = py::array_t<float>({size});
  auto ret_r = ret.mutable_unchecked<1>();

  auto ids1_r = ids1.unchecked<1>();
  auto ids2_r = ids2.unchecked<1>();

  std::unordered_map<int, float> true_count_by_id1 = std::unordered_map<int, float>{};
  std::unordered_map<int, float> total_count_by_id1 = std::unordered_map<int, float>{};
  std::unordered_map<int, float> true_count_by_id2 = std::unordered_map<int, float>{};
  std::unordered_map<int, float> total_count_by_id2 = std::unordered_map<int, float>{};
  std::unordered_map<int64_t, float> true_count_by_id12 = std::unordered_map<int64_t, float>{};
  std::unordered_map<int64_t, float> total_count_by_id12 = std::unordered_map<int64_t, float>{};
  float all_true_count = 0;
  float all_total_count = 0;

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = targets_r(row_idx);
    int id1 = ids1_r(row_idx);
    int id2 = ids2_r(row_idx);
    if(!std::isnan(target)){
      if(id1 >= 0) {
        if(true_count_by_id1.find(id1) != true_count_by_id1.end()){
          true_count_by_id1[id1] += target;
          total_count_by_id1[id1] += 1;
        } else {
          true_count_by_id1[id1] = target;
          total_count_by_id1[id1] = 1;
        }
      }

      if(id2 >= 0) {
        if(true_count_by_id2.find(id2) != true_count_by_id2.end()){
          true_count_by_id2[id2] += target;
          total_count_by_id2[id2] += 1;
        } else {
          true_count_by_id2[id2] = target;
          total_count_by_id2[id2] = 1;
        }
      }

      if(id1 >= 0 && id2 >=0) {
        int64_t id12 = ((int64_t)id1 << 32) + (int64_t)id2;
        if(true_count_by_id12.find(id12) != true_count_by_id12.end()){
          true_count_by_id12[id12] += target;
          total_count_by_id12[id12] += 1;
        } else {
          true_count_by_id12[id12] = target;
          total_count_by_id12[id12] = 1;
        }
      }

      all_true_count += target;
      all_total_count += 1;
    }
  }
  float all_mean = all_true_count / all_total_count;

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = targets_r(row_idx);
    int id1 = ids1_r(row_idx);
    int id2 = ids2_r(row_idx);
    float true_count_of_row = 0;
    float total_count_of_row = 0;

    if (id1 >= 0) {
      if(!std::isnan(target)){
        true_count_of_row += k1 * (true_count_by_id1[id1] - target) / (total_count_by_id1[id1] - 1);
        total_count_of_row += k1;
      } else {
        true_count_of_row += k1 * true_count_by_id1[id1] / total_count_by_id1[id1];
        total_count_of_row += k1;
      }
    }

    if (id2 >= 0) {
      if(!std::isnan(target)){
        true_count_of_row += k2 * (true_count_by_id2[id2] - target) / (total_count_by_id2[id2] - 1);
        total_count_of_row += k2;
      } else {
        true_count_of_row += k2 * true_count_by_id2[id2] / total_count_by_id2[id2];
        total_count_of_row += k2;
      }
    }

    if (id1 >= 0 && id2 >= 0) {
      int64_t id12 = ((int64_t)id1 << 32) + (int64_t)id2;
      if(!std::isnan(target)){
        true_count_of_row += true_count_by_id12[id12] - target;
        total_count_of_row += total_count_by_id12[id12] - 1;
      } else {
        true_count_of_row += true_count_by_id12[id12];
        total_count_of_row += total_count_by_id12[id12];
      }
    }

    if(total_count_of_row > 0 || k0 > 0) {
      ret_r(row_idx) = (true_count_of_row + k0 * all_mean) / (total_count_of_row + k0);
    } else {
      ret_r(row_idx) = all_mean;
    }
  }
  return ret;
}

py::array_t<float> FactorizedTargetEncoder::temporal_encode(
  py::array_t<float> targets,
  py::array_t<int> ids1,
  py::array_t<int> ids2,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  float k0, float k1, float k2
){
  const int size = targets.shape(0);
  auto targets_r = targets.unchecked<1>();

  py::array_t<float> ret = py::array_t<float>({size});
  auto ret_r = ret.mutable_unchecked<1>();

  auto ids1_r = ids1.unchecked<1>();
  auto ids2_r = ids2.unchecked<1>();

  float total_sum = 0;
  float total_count = 0;
  for (int row_idx = 0; row_idx < size; row_idx++) {
    float target = targets_r(row_idx);
    if(!std::isnan(target)){
      total_sum += target;
      total_count++;
    }
  }
  float total_mean = total_sum / total_count;

  auto merged_sorted_index_ = merge_sorted_index(
    time_data, time_data, sorted_index, sorted_index, true
  );
  py::array_t<int> merged_is_src = merged_sorted_index_.first;
  py::array_t<int> merged_sorted_index = merged_sorted_index_.second;
  auto merged_is_src_r = merged_is_src.unchecked<1>();
  auto merged_sorted_index_r = merged_sorted_index.unchecked<1>();
  
  std::unordered_map<int, float> true_count_by_id1 = std::unordered_map<int, float>{};
  std::unordered_map<int, float> total_count_by_id1 = std::unordered_map<int, float>{};
  std::unordered_map<int, float> true_count_by_id2 = std::unordered_map<int, float>{};
  std::unordered_map<int, float> total_count_by_id2 = std::unordered_map<int, float>{};
  std::unordered_map<int64_t, float> true_count_by_id12 = std::unordered_map<int64_t, float>{};
  std::unordered_map<int64_t, float> total_count_by_id12 = std::unordered_map<int64_t, float>{};

  for(int merged_idx = 0; merged_idx < merged_is_src.size(); merged_idx++){
    int is_src = merged_is_src_r(merged_idx);
    int src_or_dst_index = merged_sorted_index_r(merged_idx);
    int id1 = ids1_r(src_or_dst_index);
    int id2 = ids2_r(src_or_dst_index);
    float target = targets_r(src_or_dst_index);
    if(is_src) {
      float true_count_of_row = 0;
      float total_count_of_row = 0;
    
      if (id1 >= 0) {
        true_count_of_row += k1 * true_count_by_id1[id1] / total_count_by_id1[id1];
        total_count_of_row += k1;
      }

      if (id2 >= 0) {
        true_count_of_row += k2 * true_count_by_id2[id2] / total_count_by_id2[id2];
        total_count_of_row += k2;
      }

      if (id1 >= 0 && id2 >= 0) {
        int64_t id12 = ((int64_t)id1 << 32) + (int64_t)id2;
        true_count_of_row += true_count_by_id12[id12];
        total_count_of_row += total_count_by_id12[id12];
      }

      if(total_count_of_row > 0 || k0 > 0) {
        ret_r(src_or_dst_index) = (true_count_of_row + k0 * total_mean) / (total_count_of_row + k0);
      } else {
        ret_r(src_or_dst_index) = total_mean;
      }
    } else {
      if(!std::isnan(target)){
        if(id1 >= 0) {
          if(true_count_by_id1.find(id1) != true_count_by_id1.end()){
            true_count_by_id1[id1] += target;
            total_count_by_id1[id1] += 1;
          } else {
            true_count_by_id1[id1] = target;
            total_count_by_id1[id1] = 1;
          }
        }

        if(id2 >= 0) {
          if(true_count_by_id2.find(id2) != true_count_by_id2.end()){
            true_count_by_id2[id2] += target;
            total_count_by_id2[id2] += 1;
          } else {
            true_count_by_id2[id2] = target;
            total_count_by_id2[id2] = 1;
          }
        }

        if(id1 >= 0 && id2 >=0) {
          int64_t id12 = ((int64_t)id1 << 32) + (int64_t)id2;
          if(true_count_by_id12.find(id12) != true_count_by_id12.end()){
            true_count_by_id12[id12] += target;
            total_count_by_id12[id12] += 1;
          } else {
            true_count_by_id12[id12] = target;
            total_count_by_id12[id12] = 1;
          }
        }
      }
    }
  }
  return ret;
}

void InitFactorizedTargetEncoder(pybind11::module& m){
  m.doc() = "factorized target encoder made by pybind11"; // optional

  py::class_<FactorizedTargetEncoder>(m, "FactorizedTargetEncoder")
    .def(py::init<>())
    .def("encode", &FactorizedTargetEncoder::encode)
    .def("temporal_encode", &FactorizedTargetEncoder::temporal_encode);
}
