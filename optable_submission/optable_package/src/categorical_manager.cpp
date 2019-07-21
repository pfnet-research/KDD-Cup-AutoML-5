#include "categorical_manager.h"

CategoricalManager::CategoricalManager(
  std::vector<std::string> categorical_string_values
){
  int current_categorical_id = 0;
  std::unordered_map<std::string, int> categorical_ids = std::unordered_map<std::string, int>{};
  this->row_num = categorical_string_values.size();
  this->categorical_values = std::vector<int>(this->row_num);
  this->categorical_frequency = std::vector<int>{};
  this->has_null = false;
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    std::string cat_str = categorical_string_values[row_idx];
    if (false /*cat_str == ""*/) {
      this->has_null = true;
      this->categorical_values[row_idx] = -1;
    } else {
      if (categorical_ids.find(cat_str) == categorical_ids.end()) {
        categorical_ids[cat_str] = current_categorical_id;
        this->categorical_frequency.push_back(1);
        this->categorical_values[row_idx] = current_categorical_id;
        current_categorical_id++;
      } else {
        int cat_id = categorical_ids[cat_str];
        this->categorical_frequency[cat_id]++;
        this->categorical_values[row_idx] = cat_id;
      }
    }
  }
  
  this->unique_num = current_categorical_id;
  this->sorted_categorical = std::vector<int>(this->unique_num);
  for(int cat_id = 0; cat_id < this->unique_num; cat_id++) {
    this->sorted_categorical[cat_id] = cat_id;
  }
  
  std::sort(this->sorted_categorical.begin(),
            this->sorted_categorical.end(),
            [this](
              int cat_id1, int cat_id2
            ) {
              return this->categorical_frequency[cat_id1] > this->categorical_frequency[cat_id2];
            });
}

std::unique_ptr<std::vector<int>> CategoricalManager::label_(){
  std::unique_ptr<std::vector<int>> label_vec
    = std::make_unique<std::vector<int>>(this->row_num);
    
  for (int row_idx = 0; row_idx < this->row_num; row_idx++) {
    (*label_vec)[row_idx] = this->categorical_values[row_idx];
  }
  return std::move(label_vec);
}

py::array_t<int> CategoricalManager::label(){
  std::unique_ptr<std::vector<int>> ret_vec;
  {
    py::gil_scoped_release release;
    ret_vec = this->label_();
  }
  py::array_t<int> ret = py::array_t<int>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::unique_ptr<std::vector<float>> CategoricalManager::frequency_(){
  std::unique_ptr<std::vector<float>> frequency_vec
    = std::make_unique<std::vector<float>>(this->row_num);
  
  for (int row_idx = 0; row_idx < this->row_num; row_idx++) {
    int cat_id = this->categorical_values[row_idx];
    if(cat_id < 0){
      (*frequency_vec)[row_idx] = std::nanf("");
    } else {
      (*frequency_vec)[row_idx] = this->categorical_frequency[cat_id];
    }
  }
  return std::move(frequency_vec);
}

py::array_t<float> CategoricalManager::frequency(){
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    ret_vec = this->frequency_();
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::unique_ptr<std::vector<int>> CategoricalManager::is_null_(){
  std::unique_ptr<std::vector<int>> is_null_vec
    = std::make_unique<std::vector<int>>(this->row_num);
    
  for (int row_idx = 0; row_idx < this->row_num; row_idx++) {
    int cat_id = this->categorical_values[row_idx];
    if (cat_id < 0) {
      (*is_null_vec)[row_idx] = 1;
    } else {
      (*is_null_vec)[row_idx] = 0;
    }
  }
  return std::move(is_null_vec);
}

py::array_t<int> CategoricalManager::is_null(){
  std::unique_ptr<std::vector<int>> ret_vec;
  {
    py::gil_scoped_release release;
    ret_vec = this->is_null_();
  }
  py::array_t<int> ret = py::array_t<int>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::vector<std::pair<int, int>> CategoricalManager::most_common(int n){
  std::vector<std::pair<int, int>> ret = {};
  if(n > this->unique_num) {n = this->unique_num;}
  for(int cat_idx = 0; cat_idx < n; cat_idx++) {
    int cat_id = this->sorted_categorical[cat_idx];
    ret.push_back({cat_id, this->categorical_frequency[cat_id]});
  }
  return ret;
}

std::unique_ptr<std::vector<int>> CategoricalManager::is_(int objective_cat_id){
  std::unique_ptr<std::vector<int>> is_vec
    = std::make_unique<std::vector<int>>(this->row_num);
    
  for (int row_idx = 0; row_idx < this->row_num; row_idx++) {
    if (this->categorical_values[row_idx] == objective_cat_id) {
      (*is_vec)[row_idx] = 1;
    } else {
      (*is_vec)[row_idx] = 0;
    }
  }
  return std::move(is_vec);
}

py::array_t<int> CategoricalManager::is(int objective_cat_id) {
  std::unique_ptr<std::vector<int>> ret_vec;
  {
    py::gil_scoped_release release;
    ret_vec = this->is_(objective_cat_id);
  }
  py::array_t<int> ret = py::array_t<int>(ret_vec->size(), ret_vec->data());
  return ret;
}

py::array_t<int> CategoricalManager::sequential_count_encoding(py::array_t<int> sorted_index, int neighbor){
  auto sorted_index_r = sorted_index.unchecked<1>();

  py::array_t<int> ret = py::array_t<int>({this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();

  std::vector<int> count_by_id(this->unique_num, 0);
  std::vector<int> count_by_id_neighbor_ago(this->unique_num, 0);
  for(int index_of_sorted_index = 0; index_of_sorted_index < this->row_num; index_of_sorted_index++){
    int row_idx = sorted_index_r(index_of_sorted_index);
    int cat_id = this->categorical_values[row_idx];
    count_by_id[cat_id]++;
    if(index_of_sorted_index >= neighbor){
      int row_idx_neighbor_ago = sorted_index_r(index_of_sorted_index - neighbor);
      int cat_id_neighbor_ago = this->categorical_values[row_idx_neighbor_ago];
      count_by_id_neighbor_ago[cat_id_neighbor_ago]++;
    }
    ret_r[row_idx] = count_by_id[cat_id] - count_by_id_neighbor_ago[cat_id];
  }
  return ret;
}

std::unique_ptr<std::vector<float>> CategoricalManager::target_encode_with_dst_induces_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> dst_induces,
  float k
){
  const int size = targets->size();

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

  std::vector<float> sum_by_id(this->unique_num, 0);
  std::vector<float> count_by_id(this->unique_num, 0);
  for (int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    int id = this->categorical_values[dst_idx];
    if(!std::isnan(target) && id >= 0){
      sum_by_id[id] += target;
      count_by_id[id] += 1;
    }
  }
  
  for (int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    int id = this->categorical_values[dst_idx];
    if (id < 0) {
      (*ret)[row_idx] = std::nanf("");
    } else if(!std::isnan(target)) {
      if (count_by_id[id] - target > 0) {
        (*ret)[row_idx] = (sum_by_id[id] - target + total_mean * k) / (count_by_id[id] - 1 + k);
      } else {
        (*ret)[row_idx] = total_mean;
      }
    } else {
      if (count_by_id[id] > 0) {
        (*ret)[row_idx] = (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
      } else {
        (*ret)[row_idx] = total_mean;
      }
    }
  }

  return std::move(ret);
}

std::unique_ptr<std::vector<float>> CategoricalManager::temporal_target_encode_with_dst_induces_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<float>> time_data,
  std::shared_ptr<std::vector<int>> sorted_index,
  float k
){
  const int size = targets->size();

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

  std::vector<float> sum_by_id(this->unique_num);
  std::vector<float> count_by_id(this->unique_num);

  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    float target = (*targets)[src_or_dst_index];
    int dst_idx = (*dst_induces)[src_or_dst_index];
    int id = this->categorical_values[dst_idx];
    if(is_src) {
      if (id < 0) {
        (*ret)[src_or_dst_index] = std::nanf("");
      } else {
        if (count_by_id[id] > 0) {
          (*ret)[src_or_dst_index] = (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
        } else {
          (*ret)[src_or_dst_index] = total_mean;
        }
      }
    } else {
      if(!std::isnan(target) && id >= 0){
        sum_by_id[id] += target;
        count_by_id[id] += 1;
      }
    }
  }

  return std::move(ret);
}

std::unique_ptr<std::vector<bool>> CategoricalManager::calculate_adversarial_valid(
  std::shared_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
  std::shared_ptr<std::vector<int64_t>> adversarial_total_count
) {
  const int size = dst_induces->size();
  std::vector<int64_t> adversarial_true_count_by_id(this->unique_num, 0);
  std::vector<int64_t> adversarial_total_count_by_id(this->unique_num, 0);
  for(int row_idx = 0; row_idx < size; row_idx++) {
    int64_t adversarial_true_count_of_row = (*adversarial_true_count)[row_idx];
    int64_t adversarial_total_count_of_row = (*adversarial_total_count)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    if(dst_idx >= 0 and dst_idx < this->row_num){
      int cat_id = this->categorical_values[dst_idx];
      adversarial_true_count_by_id[cat_id] += adversarial_true_count_of_row;
      adversarial_total_count_by_id[cat_id] += adversarial_total_count_of_row;
    }
  }
  
  std::unique_ptr<std::vector<bool>> adverarial_valid
    = std::make_unique<std::vector<bool>>(this->unique_num);
  for(int cat_id = 0; cat_id < this->unique_num; cat_id++) {
    float ratio
      = (float)adversarial_true_count_by_id[cat_id]
      / (float)adversarial_total_count_by_id[cat_id];
    if(ratio > 0.99 || ratio < 0.01) {
      (*adverarial_valid)[cat_id] = false;
    } else {
      (*adverarial_valid)[cat_id] = true;
    }
  }
  
  return std::move(adverarial_valid);
}

std::unique_ptr<std::vector<float>> CategoricalManager::target_encode_with_adversarial_regularization_(
  std::unique_ptr<std::vector<float>> targets,
  std::shared_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
  std::shared_ptr<std::vector<int64_t>> adversarial_total_count,
  float k
){
  std::unique_ptr<std::vector<bool>> adverarial_valid
    = this->calculate_adversarial_valid(dst_induces, adversarial_true_count, adversarial_total_count);

  const int size = targets->size();

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

  std::vector<float> sum_by_id(this->unique_num, 0);
  std::vector<float> count_by_id(this->unique_num, 0);
  for (int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    int id = this->categorical_values[dst_idx];
    if(!(*adverarial_valid)[id]){continue;}
    if(!std::isnan(target) && id >= 0){
      sum_by_id[id] += target;
      count_by_id[id] += 1;
    }
  }
  
  for (int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    int id = this->categorical_values[dst_idx];
    if (id < 0) {
      (*ret)[row_idx] = std::nanf("");
    } else if(!std::isnan(target)) {
      if (count_by_id[id] - target > 0) {
        (*ret)[row_idx] = (sum_by_id[id] - target + total_mean * k) / (count_by_id[id] - 1 + k);
      } else {
        (*ret)[row_idx] = total_mean;
      }
    } else {
      if (count_by_id[id] > 0) {
        (*ret)[row_idx] = (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
      } else {
        (*ret)[row_idx] = total_mean;
      }
    }
  }

  return std::move(ret);
}

std::unique_ptr<std::vector<float>> CategoricalManager::temporal_target_encode_with_adversarial_regularization_(
  std::unique_ptr<std::vector<float>> targets,
  std::shared_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<float>> time_data,
  std::shared_ptr<std::vector<int>> sorted_index,
  std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
  std::shared_ptr<std::vector<int64_t>> adversarial_total_count,
  float k
){
  std::unique_ptr<std::vector<bool>> adverarial_valid
    = this->calculate_adversarial_valid(dst_induces, adversarial_true_count, adversarial_total_count);

  const int size = targets->size();

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

  std::vector<float> sum_by_id(this->unique_num);
  std::vector<float> count_by_id(this->unique_num);

  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    float target = (*targets)[src_or_dst_index];
    int dst_idx = (*dst_induces)[src_or_dst_index];
    int id = this->categorical_values[dst_idx];
    if(is_src) {
      if (id < 0) {
        (*ret)[src_or_dst_index] = std::nanf("");
      } else {
        if (count_by_id[id] > 0) {
          (*ret)[src_or_dst_index] = (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
        } else {
          (*ret)[src_or_dst_index] = total_mean;
        }
      }
    } else {
      if(!(*adverarial_valid)[id]){continue;}
      if(!std::isnan(target) && id >= 0){
        sum_by_id[id] += target;
        count_by_id[id] += 1;
      }
    }
  }

  return std::move(ret);
}

py::array_t<float> CategoricalManager::target_encode_with_dst_induces(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  float k
) {
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;
  
  py::buffer_info dst_induces_buf = dst_induces.request();
  int *dst_induces_ptr = (int *) dst_induces_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> dst_induces_vec
      = std::make_unique<std::vector<int>>(dst_induces_ptr, dst_induces_ptr + size);
      
    ret_vec = this->target_encode_with_dst_induces_(
      std::move(targets_vec), std::move(dst_induces_vec), k);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

py::array_t<float> CategoricalManager::temporal_target_encode_with_dst_induces(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  float k
){
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;

  py::buffer_info dst_induces_buf = dst_induces.request();
  int *dst_induces_ptr = (int *) dst_induces_buf.ptr;

  py::buffer_info time_data_buf = time_data.request();
  float *time_data_ptr = (float *) time_data_buf.ptr;
  
  py::buffer_info sorted_index_buf = sorted_index.request();
  int *sorted_index_ptr = (int *) sorted_index_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> dst_induces_vec
      = std::make_unique<std::vector<int>>(dst_induces_ptr, dst_induces_ptr + size);
    std::unique_ptr<std::vector<float>> time_data_vec
      = std::make_unique<std::vector<float>>(time_data_ptr, time_data_ptr + size);
    std::unique_ptr<std::vector<int>> sorted_index_vec
      = std::make_unique<std::vector<int>>(sorted_index_ptr, sorted_index_ptr + size);
    
    ret_vec = this->temporal_target_encode_with_dst_induces_(
      std::move(targets_vec), std::move(dst_induces_vec),
      std::move(time_data_vec), std::move(sorted_index_vec), k);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

py::array_t<float> CategoricalManager::target_encode_with_adversarial_regularization(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  py::array_t<int64_t> adversarial_true_count,
  py::array_t<int64_t> adversarial_total_count,
  float k
) {
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;
  
  py::buffer_info dst_induces_buf = dst_induces.request();
  int *dst_induces_ptr = (int *) dst_induces_buf.ptr;

  py::buffer_info adversarial_true_count_buf = adversarial_true_count.request();
  int64_t* adversarial_true_count_ptr = (int64_t*) adversarial_true_count_buf.ptr;

  py::buffer_info adversarial_total_count_buf = adversarial_total_count.request();
  int64_t* adversarial_total_count_ptr = (int64_t*)adversarial_total_count_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> dst_induces_vec
      = std::make_unique<std::vector<int>>(dst_induces_ptr, dst_induces_ptr + size);
    std::unique_ptr<std::vector<int64_t>> adversarial_true_count_vec
      = std::make_unique<std::vector<int64_t>>(adversarial_true_count_ptr, adversarial_true_count_ptr + size);
    std::unique_ptr<std::vector<int64_t>> adversarial_total_count_vec
      = std::make_unique<std::vector<int64_t>>(adversarial_total_count_ptr, adversarial_total_count_ptr + size);
    
    ret_vec = this->target_encode_with_adversarial_regularization_(
      std::move(targets_vec), std::move(dst_induces_vec),
      std::move(adversarial_true_count_vec), std::move(adversarial_total_count_vec), k);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

py::array_t<float> CategoricalManager::temporal_target_encode_with_adversarial_regularization(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  py::array_t<int64_t> adversarial_true_count,
  py::array_t<int64_t> adversarial_total_count,
  float k
){
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;

  py::buffer_info dst_induces_buf = dst_induces.request();
  int *dst_induces_ptr = (int *) dst_induces_buf.ptr;

  py::buffer_info time_data_buf = time_data.request();
  float *time_data_ptr = (float *) time_data_buf.ptr;
  
  py::buffer_info sorted_index_buf = sorted_index.request();
  int *sorted_index_ptr = (int *) sorted_index_buf.ptr;
  
  py::buffer_info adversarial_true_count_buf = adversarial_true_count.request();
  int64_t *adversarial_true_count_ptr = (int64_t *)adversarial_true_count_buf.ptr;
  
  py::buffer_info adversarial_total_count_buf = adversarial_total_count.request();
  int64_t *adversarial_total_count_ptr = (int64_t *)adversarial_total_count_buf.ptr;

  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> dst_induces_vec
      = std::make_unique<std::vector<int>>(dst_induces_ptr, dst_induces_ptr + size);
    std::unique_ptr<std::vector<float>> time_data_vec
      = std::make_unique<std::vector<float>>(time_data_ptr, time_data_ptr + size);
    std::unique_ptr<std::vector<int>> sorted_index_vec
      = std::make_unique<std::vector<int>>(sorted_index_ptr, sorted_index_ptr + size);
    std::unique_ptr<std::vector<int64_t>> adversarial_true_count_vec
      = std::make_unique<std::vector<int64_t>>(adversarial_true_count_ptr, adversarial_true_count_ptr + size);
    std::unique_ptr<std::vector<int64_t>> adversarial_total_count_vec
      = std::make_unique<std::vector<int64_t>>(adversarial_total_count_ptr, adversarial_total_count_ptr + size);
    
    ret_vec = this->temporal_target_encode_with_adversarial_regularization_(
      std::move(targets_vec), std::move(dst_induces_vec),
      std::move(time_data_vec), std::move(sorted_index_vec), 
      std::move(adversarial_true_count_vec), std::move(adversarial_total_count_vec),
      k);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::unique_ptr<std::vector<float>> CategoricalManager::exponential_moving_target_encode_with_dst_induces_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<float>> time_data,
  std::shared_ptr<std::vector<int>> sorted_index,
  float k,
  float beta
) {
  const int size = targets->size();

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

  std::vector<float> sum_by_id(this->unique_num);
  std::vector<float> count_by_id(this->unique_num);
  std::vector<float> previous_time(this->unique_num, std::nanf(""));

  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    float src_or_dst_time = (*time_data)[src_or_dst_index];
    float target = (*targets)[src_or_dst_index];
    int dst_idx = (*dst_induces)[src_or_dst_index];
    int id = this->categorical_values[dst_idx];
    if(is_src) {
      if (id < 0) {
        (*ret)[src_or_dst_index] = std::nanf("");
      } else {
        if (count_by_id[id] > 0) {
          float time_progress = src_or_dst_time - previous_time[id];
          float weight = std::exp(-beta * time_progress);
          sum_by_id[id] *= weight;
          count_by_id[id] *= weight;
          previous_time[id] = src_or_dst_time;

          (*ret)[src_or_dst_index] = (sum_by_id[id] + total_mean * k) / (count_by_id[id] + k);
        } else {
          (*ret)[src_or_dst_index] = total_mean;
        }
      }
    } else {
      if(!std::isnan(target) && id >= 0){
        if(!std::isnan(previous_time[id])){
          float time_progress = src_or_dst_time - previous_time[id];
          float weight = std::exp(-beta * time_progress);
          sum_by_id[id] *= weight;
          count_by_id[id] *= weight;
          previous_time[id] = src_or_dst_time;

          sum_by_id[id] += target;
          count_by_id[id] += 1;
        } else {
          if(!std::isnan(src_or_dst_time)){
            previous_time[id] = src_or_dst_time;

            sum_by_id[id] += target;
            count_by_id[id] += 1;
          }
        }
      }
    }
  }

  return std::move(ret);
}

py::array_t<float> CategoricalManager::exponential_moving_temporal_target_encode_with_dst_induces(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  float k,
  float beta
){
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;

  py::buffer_info dst_induces_buf = dst_induces.request();
  int *dst_induces_ptr = (int *) dst_induces_buf.ptr;

  py::buffer_info time_data_buf = time_data.request();
  float *time_data_ptr = (float *) time_data_buf.ptr;
  
  py::buffer_info sorted_index_buf = sorted_index.request();
  int *sorted_index_ptr = (int *) sorted_index_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<int>> dst_induces_vec
      = std::make_unique<std::vector<int>>(dst_induces_ptr, dst_induces_ptr + size);
    std::unique_ptr<std::vector<float>> time_data_vec
      = std::make_unique<std::vector<float>>(time_data_ptr, time_data_ptr + size);
    std::unique_ptr<std::vector<int>> sorted_index_vec
      = std::make_unique<std::vector<int>>(sorted_index_ptr, sorted_index_ptr + size);
    
    ret_vec = this->exponential_moving_target_encode_with_dst_induces_(
      std::move(targets_vec), std::move(dst_induces_vec),
      std::move(time_data_vec), std::move(sorted_index_vec), k, beta);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}


void InitCategoricalManager(pybind11::module& m){
  m.doc() = "multi categorical made by pybind11"; // optional
  py::class_<CategoricalManager>(m, "CategoricalManager")
    .def_readonly("row_num", &CategoricalManager::row_num)
    .def_readonly("unique_num", &CategoricalManager::unique_num)
    .def_readonly("has_null", &CategoricalManager::has_null)
    .def("label", &CategoricalManager::label)
    .def("is_null", &CategoricalManager::is_null)
    .def("frequency", &CategoricalManager::frequency)
    .def("most_common", &CategoricalManager::most_common)
    .def("is_array", &CategoricalManager::is)
    .def("sequential_count_encoding",
      &CategoricalManager::sequential_count_encoding)
    .def("target_encode_with_dst_induces",
      &CategoricalManager::target_encode_with_dst_induces)
    .def("temporal_target_encode_with_dst_induces",
      &CategoricalManager::temporal_target_encode_with_dst_induces)
    .def("exponential_moving_temporal_target_encode_with_dst_induces",
      &CategoricalManager::exponential_moving_temporal_target_encode_with_dst_induces)
    .def("target_encode_with_adversarial_regularization",
      &CategoricalManager::target_encode_with_adversarial_regularization)
    .def("temporal_target_encode_with_adversarial_regularization",
      &CategoricalManager::temporal_target_encode_with_adversarial_regularization)
    .def(py::init<std::vector<std::string>>());
}
