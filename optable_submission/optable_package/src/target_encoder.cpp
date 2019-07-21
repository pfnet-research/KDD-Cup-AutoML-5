#include "target_encoder.h"

std::unique_ptr<std::vector<float>> TargetEncoder::encode_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> ids,
  int k
){
   const int size = ids->size();
   std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);

   float total_sum = 0;
   float total_count = 0;
   for (int row_idx = 0; row_idx < size; row_idx++) {
     float target = (*targets)[row_idx];
     if(!std::isnan(target)){
       total_sum += target;
       total_count++;
     }
   }
   float total_mean = total_sum / total_count;
   
   std::unordered_map<int, float> sum_by_id = std::unordered_map<int, float>{};
   std::unordered_map<int, float> count_by_id = std::unordered_map<int, float>{};
   for (int row_idx = 0; row_idx < size; row_idx++) {
     int id = (*ids)[row_idx];
     float target = (*targets)[row_idx];
     if(!std::isnan(target) && id != -1){
       if (sum_by_id.find(id) != sum_by_id.end()) {
         sum_by_id[id] += target;
         count_by_id[id] += 1;
       } else {
         sum_by_id[id] = target;
         count_by_id[id] = 1;
       }
     }
   }
   
   for (int row_idx = 0; row_idx < size; row_idx++) {
     int id = (*ids)[row_idx];
     float target = (*targets)[row_idx];
     if (id == -1) {
       (*ret)[row_idx] = std::nanf("");
     }else if (sum_by_id.find(id) != sum_by_id.end()) {
       if(!std::isnan(target)){
         (*ret)[row_idx] = (sum_by_id[id] - target + total_mean * k) / (count_by_id[id] - 1 + k);
       } else {
         (*ret)[row_idx] = (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
       }
     } else {
       (*ret)[row_idx] = total_mean;
     }
   }
   
   return std::move(ret);
 }
 
py::array_t<float> TargetEncoder::encode(
  py::array_t<float> targets,
  py::array_t<int> ids,
  int k // k=5, 25がKDD2014で使われていた。
){
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;
  
  py::buffer_info ids_buf = ids.request();
  int * ids_ptr = (int *) ids_buf.ptr;

  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> ids_vec
      = std::make_unique<std::vector<int>>(ids_ptr, ids_ptr + size);
    
    ret_vec = this->encode_(std::move(targets_vec), std::move(ids_vec), k);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::unique_ptr<std::vector<float>> TargetEncoder::not_loo_encode_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> ids,
  int k
){
   const int size = ids->size();
   std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);

   float total_sum = 0;
   float total_count = 0;
   for (int row_idx = 0; row_idx < size; row_idx++) {
     float target = (*targets)[row_idx];
     if(!std::isnan(target)){
       total_sum += target;
       total_count++;
     }
   }
   float total_mean = total_sum / total_count;
   
   std::unordered_map<int, float> sum_by_id = std::unordered_map<int, float>{};
   std::unordered_map<int, float> count_by_id = std::unordered_map<int, float>{};
   for (int row_idx = 0; row_idx < size; row_idx++) {
     int id = (*ids)[row_idx];
     float target = (*targets)[row_idx];
     if(!std::isnan(target) && id != -1){
       if (sum_by_id.find(id) != sum_by_id.end()) {
         sum_by_id[id] += target;
         count_by_id[id] += 1;
       } else {
         sum_by_id[id] = target;
         count_by_id[id] = 1;
       }
     }
   }
   
   for (int row_idx = 0; row_idx < size; row_idx++) {
     int id = (*ids)[row_idx];
     float target = (*targets)[row_idx];
     if (id == -1) {
       (*ret)[row_idx] = std::nanf("");
     }else if (sum_by_id.find(id) != sum_by_id.end()) {
       (*ret)[row_idx] = (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
     } else {
       (*ret)[row_idx] = total_mean;
     }
   }
   
   return std::move(ret);
 }
 
py::array_t<float> TargetEncoder::not_loo_encode(
  py::array_t<float> targets,
  py::array_t<int> ids,
  int k // k=5, 25がKDD2014で使われていた。
){
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;
  
  py::buffer_info ids_buf = ids.request();
  int * ids_ptr = (int *) ids_buf.ptr;

  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> ids_vec
      = std::make_unique<std::vector<int>>(ids_ptr, ids_ptr + size);
    
    ret_vec = this->not_loo_encode_(std::move(targets_vec), std::move(ids_vec), k);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::unique_ptr<std::vector<float>> TargetEncoder::temporal_encode_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> ids,
  std::shared_ptr<std::vector<float>> time_data,
  std::shared_ptr<std::vector<int>> sorted_index,
  int k
) {
  const int size = ids->size();
  
  std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);

  float total_sum = 0;
  float total_count = 0;
  for (int idx = 0; idx < size; idx++) {
    if(!std::isnan((*targets)[idx])){
      total_sum += (*targets)[idx];
      total_count++;
    }
  }
  float total_mean = total_sum / total_count;

  auto merged_sorted_index_ = merge_sorted_index_(
    time_data, time_data,
    sorted_index, sorted_index, true
  );
  std::unique_ptr<std::vector<int>> merged_is_src = std::move(merged_sorted_index_.first);
  std::unique_ptr<std::vector<int>> merged_sorted_index = std::move(merged_sorted_index_.second);

  std::unordered_map<int, float> sum_by_id = std::unordered_map<int, float>{};
  std::unordered_map<int, float> count_by_id = std::unordered_map<int, float>{};
  
  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    int id = (*ids)[src_or_dst_index];
    float target = (*targets)[src_or_dst_index];
    if(is_src) {
      if(id == -1) {
        (*ret)[src_or_dst_index] = std::nanf("");
      } else if(sum_by_id.find(id) == sum_by_id.end()) {
        (*ret)[src_or_dst_index] = total_mean;
      } else {
        (*ret)[src_or_dst_index] =
          (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
      }
    } else {
      if(id != -1 && !std::isnan(target)) {
        if(sum_by_id.find(id) == sum_by_id.end()) {
          sum_by_id[id] = target;
          count_by_id[id] = 1;
        } else {
          sum_by_id[id] += target;
          count_by_id[id] ++;
        }
      }
    }
  }

  return std::move(ret);
}

py::array_t<float> TargetEncoder::temporal_encode(
  py::array_t<float> targets,
  py::array_t<int> ids,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  int k // k=5, 25がKDD2014で使われていた。
){
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;
  
  py::buffer_info ids_buf = ids.request();
  int *ids_ptr = (int *) ids_buf.ptr;
  
  py::buffer_info time_data_buf = time_data.request();
  float *time_data_ptr = (float*) time_data_buf.ptr;

  py::buffer_info sorted_index_buf = sorted_index.request();
  int *sorted_index_ptr = (int*) sorted_index_buf.ptr;

  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> ids_vec
      = std::make_unique<std::vector<int>>(ids_ptr, ids_ptr + size);
    std::unique_ptr<std::vector<float>> time_data_buf_vec
      = std::make_unique<std::vector<float>>(time_data_ptr, time_data_ptr+size);
    std::unique_ptr<std::vector<int>> sorted_index_vec
      = std::make_unique<std::vector<int>>(sorted_index_ptr, sorted_index_ptr+size);
    
    ret_vec = this->temporal_encode_(
        std::move(targets_vec), std::move(ids_vec),
        std::move(time_data_buf_vec), std::move(sorted_index_vec), k);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

void InitTargetEncoder(pybind11::module& m){
  m.doc() = "target aggregator made by pybind11"; // optional

  py::class_<TargetEncoder>(m, "TargetEncoder")
    .def(py::init<>())
    .def("encode", &TargetEncoder::encode)
    .def("not_loo_encode", &TargetEncoder::not_loo_encode)
    .def("temporal_encode", &TargetEncoder::temporal_encode);
}
