#include "multi_categorical_manager.h"


std::pair<std::unique_ptr<std::unordered_map<std::string, int>>, int> text_to_frequency_map(const std::string & text, const int& restrict) {
  std::unique_ptr<std::unordered_map<std::string, int>> frequency_map = std::make_unique<std::unordered_map<std::string, int>>();
  std::string::size_type pos = 0;
  std::string::size_type next_pos;
  int size = 0;
  while(pos < text.size()) {
    next_pos = text.find(",", pos);
    if (next_pos == std::string::npos) next_pos = text.size();
    if (size <= restrict) {
      std::string word = text.substr(pos, next_pos - pos);
      if(frequency_map->find(word) != frequency_map->end()) {
        (*frequency_map)[word] += 1;
      } else {
        (*frequency_map)[word] = 1;
      }
    }
    size += 1;
    pos = next_pos + 1;  
  }
  return {std::move(frequency_map), size};
}

MultiCategoricalManager::MultiCategoricalManager(
  std::vector<std::string> multi_categorical_string_values
) {
  this->row_num = (int)(multi_categorical_string_values.size());
  this->multi_categorical_values =
    std::vector<std::unique_ptr<std::unordered_map<int, int>>>(this->row_num);
  this->length_vec = std::vector<int>(this->row_num);
  std::unordered_map<std::string, int> word_to_id = std::unordered_map<std::string, int>{};
  int current_word_id = 0;
  int restrict = 100000000 / this->row_num;
  if(restrict < 10) restrict = 10;
  for (int row_idx = 0; row_idx < multi_categorical_string_values.size(); row_idx++) {
    auto freq_map_and_size = text_to_frequency_map(multi_categorical_string_values[row_idx], restrict);
    std::unique_ptr<std::unordered_map<int, int>> multi_categorical_value =
      std::make_unique<std::unordered_map<int, int>>();
    for (auto word_it = freq_map_and_size.first->begin();
         word_it != freq_map_and_size.first->end();
         word_it++) {
      if (word_to_id.find(word_it->first) == word_to_id.end()) {
        word_to_id[word_it->first] = current_word_id;
        current_word_id++;
      }
      (*multi_categorical_value)[word_to_id[word_it->first]] = word_it->second;
    }
    this->multi_categorical_values[row_idx] = std::move(multi_categorical_value);
    this->total_word_num += freq_map_and_size.second;
    this->length_vec[row_idx] = freq_map_and_size.second;
    if (freq_map_and_size.second > this->max_word_num) this->max_word_num = freq_map_and_size.second;
  }
  this->mean_word_num = (float)this->total_word_num / (float)this->row_num;
  this->unique_word_num = current_word_id;

  this->num_of_words = std::vector<float>(this->unique_word_num, 0);
  this->document_num_of_words = std::vector<float>(this->unique_word_num, 0);
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
        word_id_it != this->multi_categorical_values[row_idx]->end();
        word_id_it++) {
      int word_id = word_id_it->first;
      int word_count = word_id_it->second;
      this->num_of_words[word_id] += word_count;
      this->document_num_of_words[word_id] ++;
    }
  }

  this->words_sorted_by_num = std::vector<int>(this->unique_word_num);
  for(int word_id = 0; word_id < this->unique_word_num; word_id++) {
    this->words_sorted_by_num[word_id] = word_id;
  }
  std::sort(this->words_sorted_by_num.begin(),
            this->words_sorted_by_num.end(),
            [this](
              int word_id1, int word_id2
            ) {
              return this->document_num_of_words[word_id1] > this->document_num_of_words[word_id2];
            });

  /*
  this->tfidf_of_1word = std::vector<float>(this->unique_word_num, 0);
  for(int word_id = 0; word_id < this->unique_word_num; word_id++) {
    float num_of_word = this->num_of_words[word_id];
    float document_num_of_word = document_num_of_words[word_id];
    this->tfidf_of_1word[word_id] = 1 / num_of_word * (std::log(this->row_num) - std::log(document_num_of_word));
  }
  */

  /*
  this->tfidf_values =
    std::vector<std::unique_ptr<std::unordered_map<int, float>>>(this->row_num);
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    float norm = 0;
    for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
        word_id_it != this->multi_categorical_values[row_idx]->end();
        word_id_it++) {
      float norm_cand = this->tfidf_of_1word[word_id_it->first] * word_id_it->second;
      if (norm_cand > norm) {
        norm = norm_cand;
      }
    }
    std::unique_ptr<std::unordered_map<int, float>> tfidf_value =
      std::make_unique<std::unordered_map<int, float>>();
    for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
        word_id_it != this->multi_categorical_values[row_idx]->end();
        word_id_it++) {
      (*tfidf_value)[word_id_it->first] =
        this->tfidf_of_1word[word_id_it->first] * word_id_it->second / norm;
    }
    this->tfidf_values[row_idx] = std::move(tfidf_value);
  }
  */

  this->hash_dimension = 0;
};

/*
py::array_t<float> MultiCategoricalManager::tfidf_target_encode(
  py::array_t<float> targets,
  float k1, float k2
) {
  const int size = targets.shape(0);
  auto targets_r = targets.unchecked<1>();

  py::array_t<float> ret = py::array_t<float>({(std::size_t)(size)});
  auto ret_r = ret.mutable_unchecked<1>();

  std::unordered_map<int, float> true_count_by_words = std::unordered_map<int, float>{};
  std::unordered_map<int, float> total_count_by_words = std::unordered_map<int, float>{};

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = targets_r(row_idx);
    if(!std::isnan(target)){
      for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
          word_id_it != this->multi_categorical_values[row_idx]->end();
          word_id_it++) {
        int word_id = word_id_it->first;
        float word_tfidf = word_id_it->second * this->tfidf_of_1word[word_id];
        if(true_count_by_words.find(word_id) != true_count_by_words.end()){
          true_count_by_words[word_id] += target * word_tfidf;
          total_count_by_words[word_id] += word_tfidf;
        } else {
          true_count_by_words[word_id] = target * word_tfidf;
          total_count_by_words[word_id] = word_tfidf;
        }
      }
    }
  }

  float all_true_count_by_word = 0;
  float all_total_count_by_word = 0;
  for(auto it = true_count_by_words.begin();
      it != true_count_by_words.end();
      it++) {
    all_true_count_by_word += true_count_by_words[it->first];
    all_total_count_by_word += total_count_by_words[it->first];
  }

  float all_true_count_by_row = 0;
  float all_total_count_by_row = 0;
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = targets_r(row_idx);
    if(!std::isnan(target)){
      all_true_count_by_row += target;
      all_total_count_by_row ++;
    }
  }
  float all_true_mean_by_row = all_true_count_by_row / all_total_count_by_row;

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = targets_r(row_idx);
    float true_count_of_row = 0;
    float total_count_of_row = 0;
    if(!std::isnan(target)){
      for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
          word_id_it != this->multi_categorical_values[row_idx]->end();
          word_id_it++
      ) {
        int word_id = word_id_it->first;
        float word_tfidf = word_id_it->second * this->tfidf_of_1word[word_id];
        if (true_count_by_words.find(word_id) != true_count_by_words.end()) {
          true_count_of_row += word_tfidf * (true_count_by_words[word_id] - target * word_tfidf + k1 * all_true_count_by_word / all_total_count_by_word);
          total_count_of_row += word_tfidf * (total_count_by_words[word_id] - word_tfidf + k1);
        } else {
          true_count_of_row += word_tfidf * (k1 * all_true_count_by_word / all_total_count_by_word);
          total_count_of_row += word_tfidf * (k1);
        }
      }
    } else {
      for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
          word_id_it != this->multi_categorical_values[row_idx]->end();
          word_id_it++
      ) {
        int word_id = word_id_it->first;
        float word_tfidf = word_id_it->second * this->tfidf_of_1word[word_id];
        if(true_count_by_words.find(word_id) != true_count_by_words.end()){
          true_count_of_row += word_tfidf * (true_count_by_words[word_id] + k1 * all_true_count_by_word / all_total_count_by_word);
          total_count_of_row += word_tfidf * (total_count_by_words[word_id] + k1);
        } else {
          true_count_of_row += word_tfidf * (k1 * all_true_count_by_word / all_total_count_by_word);
          total_count_of_row += word_tfidf * (k1);
        }
      }
    }

    if(total_count_of_row > 0 || k2 > 0) {
      ret_r(row_idx) = (true_count_of_row + k2 * all_true_mean_by_row) / (total_count_of_row + k2);
    } else {
      ret_r(row_idx) = all_true_mean_by_row;
    }
  }

  return ret;
}
*/

py::array_t<float> MultiCategoricalManager::length(){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    ret_r(row_idx) = this->length_vec[row_idx];
  }
  return ret;
}

py::array_t<float> MultiCategoricalManager::nunique(){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    ret_r(row_idx) = this->multi_categorical_values[row_idx]->size();
  }
  return ret;
}

py::array_t<float> MultiCategoricalManager::duplicates(){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    float duplicates_of_row = 0;
    for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
        word_id_it != this->multi_categorical_values[row_idx]->end();
        word_id_it++
    ) {
      duplicates_of_row += word_id_it->second - 1;
    }
    ret_r(row_idx) = duplicates_of_row;
  }
  return ret;
}

py::array_t<int> MultiCategoricalManager::mode(){
  py::array_t<int> ret = py::array_t<int>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();

  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    if(this->multi_categorical_values[row_idx]->size() > 0) {
      auto max_iterator = std::max_element(
        this->multi_categorical_values[row_idx]->begin(),
        this->multi_categorical_values[row_idx]->end(),
        [](const auto &a, const auto &b) -> bool {
            return (a.second < b.second);
        }
      );
      ret_r(row_idx) = max_iterator->first;
    } else {
      ret_r(row_idx) = -1;
    }
  }
  return ret;
}

/*
py::array_t<int> MultiCategoricalManager::max_tfidf_words(){
  py::array_t<int> ret = py::array_t<int>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();

  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    if(this->tfidf_values[row_idx]->size() > 0) {
      auto max_iterator = std::max_element(
        this->tfidf_values[row_idx]->begin(),
        this->tfidf_values[row_idx]->end(),
        [](const auto &a, const auto &b) -> bool {
            return (a.second < b.second);
        }
      );
      ret_r(row_idx) = max_iterator->first;
    } else {
      ret_r(row_idx) = -1;
    }
  }
  return ret;
}
*/

/*
py::array_t<float> MultiCategoricalManager::tfidf(int idx){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  if (idx >= this->words_sorted_by_num.size()){
    return ret;
  }
  int word_id = this->words_sorted_by_num[idx];
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    if(this->tfidf_values[row_idx]->find(word_id) != this->tfidf_values[row_idx]->end()) {
      ret_r(row_idx) = (*this->tfidf_values[row_idx])[word_id];
    } else {
      ret_r(row_idx) = 0;
    }
  }
  return ret;
}
*/

py::array_t<float> MultiCategoricalManager::count(int idx){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  if (idx >= this->words_sorted_by_num.size()){
    return ret;
  }
  int word_id = this->words_sorted_by_num[idx];
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    if(this->multi_categorical_values[row_idx]->find(word_id)
       != this->multi_categorical_values[row_idx]->end()) {
      ret_r(row_idx) = (*this->multi_categorical_values[row_idx])[word_id];
    } else {
      ret_r(row_idx) = 0;
    }
  }
  return ret;
}

py::array_t<float> MultiCategoricalManager::max_count(){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    float max = 0;
    for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
        word_id_it != this->multi_categorical_values[row_idx]->end();
        word_id_it++
    ) {
      if (max < word_id_it->second) {
        max = word_id_it->second;
      }
    }
    ret_r(row_idx) = max;
  }
  return ret;
}

py::array_t<float> MultiCategoricalManager::min_count(){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    float min = std::nanf("");
    for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
        word_id_it != this->multi_categorical_values[row_idx]->end();
        word_id_it++
    ) {
      if (std::isnan(min)) {
        min = word_id_it->second;
      } else if (min > word_id_it->second ){
        min = word_id_it->second;
      }
    }
    ret_r(row_idx) = min;
  }
  return ret;
}


std::unique_ptr<std::vector<float>> MultiCategoricalManager::target_encode_(
  std::unique_ptr<std::vector<float>> targets,
  float k1, float k2
) {
  const int size = targets->size();

  std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);

  std::vector<float> true_count_by_words(this->unique_word_num, 0.0f);
  std::vector<float> total_count_by_words(this->unique_word_num, 0.0f);

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    if(!std::isnan(target)){
      for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
          word_id_it != this->multi_categorical_values[row_idx]->end();
          word_id_it++) {
        int word_id = word_id_it->first;
        int word_count = word_id_it->second;
        true_count_by_words[word_id] += target * word_count;
        total_count_by_words[word_id] += word_count;
      }
    }
  }

  float all_true_count_by_word = 0;
  float all_total_count_by_word = 0;
  for(int word_id = 0; word_id < this->unique_word_num; word_id++) {
    all_true_count_by_word += true_count_by_words[word_id];
    all_total_count_by_word += total_count_by_words[word_id];
  }
  float all_true_mean_by_word = all_true_count_by_word / all_total_count_by_word;

  float all_true_count_by_row = 0;
  float all_total_count_by_row = 0;
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    if(!std::isnan(target)){
      all_true_count_by_row += target;
      all_total_count_by_row ++;
    }
  }
  float all_true_mean_by_row = all_true_count_by_row / all_total_count_by_row;

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    float true_count_of_row = 0;
    float total_count_of_row = 0;
    if(!std::isnan(target)){
      for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
          word_id_it != this->multi_categorical_values[row_idx]->end();
          word_id_it++
      ) {
        int word_id = word_id_it->first;
        int word_count = word_id_it->second;
        true_count_of_row += word_count * (true_count_by_words[word_id] - target * word_count + k1 * all_true_mean_by_word);
        total_count_of_row += word_count * (total_count_by_words[word_id] - word_count + k1);
      }
    } else {
      for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
          word_id_it != this->multi_categorical_values[row_idx]->end();
          word_id_it++
      ) {
        int word_id = word_id_it->first;
        int word_count = word_id_it->second;
        true_count_of_row += word_count * (true_count_by_words[word_id] + k1 * all_true_mean_by_word);
        total_count_of_row += word_count * (total_count_by_words[word_id] + k1);
      }
    }

    if(total_count_of_row > 0 || k2 > 0) {
      (*ret)[row_idx] = (true_count_of_row + k2 * all_true_mean_by_row) / (total_count_of_row + k2);
    } else {
      (*ret)[row_idx] = all_true_mean_by_row;
    }
  }
  
  return std::move(ret);
}

py::array_t<float> MultiCategoricalManager::target_encode(
  py::array_t<float> targets,
  float k1, float k2
) {
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    
    ret_vec = this->target_encode_(std::move(targets_vec), k1, k2);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}


std::unique_ptr<std::vector<float>> MultiCategoricalManager::temporal_target_encode_(
  std::unique_ptr<std::vector<float>> targets,
  std::shared_ptr<std::vector<float>> time_data,
  std::shared_ptr<std::vector<int>> sorted_index,
  float k1, float k2
) {
  const int size = targets->size();

  std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);
  
  float all_true_count_by_row = 0;
  float all_total_count_by_row = 0;
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    if(!std::isnan(target)){
      all_true_count_by_row += target;
      all_total_count_by_row ++;
    }
  }
  float all_true_mean_by_row = all_true_count_by_row / all_total_count_by_row;

  std::vector<float> true_count_by_words(this->unique_word_num, 0.0f);
  std::vector<float> total_count_by_words(this->unique_word_num, 0.0f);

  float all_true_count_by_word = 0;
  float all_total_count_by_word = 0;

  auto merged_sorted_index_ = merge_sorted_index_(
    time_data, time_data,
    sorted_index, sorted_index, true
  );
  std::unique_ptr<std::vector<int>> merged_is_src
    = std::move(merged_sorted_index_.first);
  std::unique_ptr<std::vector<int>> merged_sorted_index
    = std::move(merged_sorted_index_.second);
  
  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    float target = (*targets)[src_or_dst_index];
    if(is_src) {
      float true_count_of_row = 0;
      float total_count_of_row = 0;
      for(auto word_id_it = this->multi_categorical_values[src_or_dst_index]->begin();
          word_id_it != this->multi_categorical_values[src_or_dst_index]->end();
          word_id_it++
      ) {
        int word_id = word_id_it->first;
        int word_count = word_id_it->second;
        true_count_of_row += word_count * (true_count_by_words[word_id] + k1 * all_true_count_by_word / all_total_count_by_word);
        total_count_of_row += word_count * (total_count_by_words[word_id] + k1);
      }
      if(total_count_of_row > 0 || k2 > 0) {
        (*ret)[src_or_dst_index] = (true_count_of_row + k2 * all_true_mean_by_row) / (total_count_of_row + k2);
      } else {
        (*ret)[src_or_dst_index] = all_true_mean_by_row;
      }
    } else {
      if(!std::isnan(target)){
        for(auto word_id_it = this->multi_categorical_values[src_or_dst_index]->begin();
            word_id_it != this->multi_categorical_values[src_or_dst_index]->end();
            word_id_it++) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          true_count_by_words[word_id] += target * word_count;
          total_count_by_words[word_id] += word_count;
          all_true_count_by_word += target * word_count;
          all_total_count_by_word += word_count;
        }
      }
    }
  }
  
  return std::move(ret);
}

py::array_t<float> MultiCategoricalManager::temporal_target_encode(
  py::array_t<float> targets,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  float k1, float k2
) {
  const int size = targets.size();

  py::buffer_info targets_buf = targets.request();
  float *targets_ptr = (float *) targets_buf.ptr;

  py::buffer_info time_data_buf = time_data.request();
  float *time_data_ptr = (float *) time_data_buf.ptr;
  
  py::buffer_info sorted_index_buf = sorted_index.request();
  int *sorted_index_ptr = (int *) sorted_index_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> targets_vec
      = std::make_unique<std::vector<float>>(targets_ptr, targets_ptr + size);
    std::unique_ptr<std::vector<float>> time_data_vec
      = std::make_unique<std::vector<float>>(time_data_ptr, time_data_ptr + size);
    std::unique_ptr<std::vector<int>> sorted_index_vec
      = std::make_unique<std::vector<int>>(sorted_index_ptr, sorted_index_ptr + size);
    
    ret_vec = this->temporal_target_encode_(
      std::move(targets_vec), std::move(time_data_vec), std::move(sorted_index_vec), k1, k2);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::unique_ptr<std::vector<float>> MultiCategoricalManager::target_encode_with_dst_induces_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> dst_induces,
  float k1, float k2
) {
  const int size = targets->size();

  std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);
  
  std::vector<float> true_count_by_words(this->unique_word_num, 0.0f);
  std::vector<float> total_count_by_words(this->unique_word_num, 0.0f);
  
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    if(!std::isnan(target)){
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          true_count_by_words[word_id] += target * word_count;
          total_count_by_words[word_id] += word_count;
        }
      }
    }
  }

  float all_true_count_by_word = 0;
  float all_total_count_by_word = 0;
  for(int word_id = 0; word_id < this->unique_word_num; word_id++) {
    all_true_count_by_word += true_count_by_words[word_id];
    all_total_count_by_word += total_count_by_words[word_id];
  }
  float all_true_mean_by_word = all_true_count_by_word / all_total_count_by_word;

  float all_true_count_by_row = 0;
  float all_total_count_by_row = 0;
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    if(!std::isnan(target)){
      all_true_count_by_row += target;
      all_total_count_by_row ++;
    }
  }
  float all_true_mean_by_row = all_true_count_by_row / all_total_count_by_row;

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    float true_count_of_row = 0;
    float total_count_of_row = 0;
    if(!std::isnan(target)){
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++
        ) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          true_count_of_row += word_count * (true_count_by_words[word_id] - target * word_count + k1 * all_true_mean_by_word);
          total_count_of_row += word_count * (total_count_by_words[word_id] - word_count + k1);
        }
      }
    } else {
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++
        ) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          true_count_of_row += word_count * (true_count_by_words[word_id] + k1 * all_true_mean_by_word);
          total_count_of_row += word_count * (total_count_by_words[word_id] + k1);
        }
      }
    }

    if(total_count_of_row > 0 || k2 > 0) {
      (*ret)[row_idx] = (true_count_of_row + k2 * all_true_mean_by_row) / (total_count_of_row + k2);
    } else {
      (*ret)[row_idx] = all_true_mean_by_row;
    }
  }
  
  return std::move(ret);
}

py::array_t<float> MultiCategoricalManager::target_encode_with_dst_induces(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  float k1, float k2
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
      std::move(targets_vec), std::move(dst_induces_vec), k1, k2);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

std::unique_ptr<std::vector<float>> MultiCategoricalManager::temporal_target_encode_with_dst_induces_(
  std::unique_ptr<std::vector<float>> targets,
  std::unique_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<float>> time_data,
  std::shared_ptr<std::vector<int>> sorted_index,
  float k1, float k2
){
  const int size = targets->size();

  std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);
  
  float all_true_count_by_row = 0;
  float all_total_count_by_row = 0;
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    if(!std::isnan(target)){
      all_true_count_by_row += target;
      all_total_count_by_row ++;
    }
  }
  float all_true_mean_by_row = all_true_count_by_row / all_total_count_by_row;

  std::vector<float> true_count_by_words(this->unique_word_num, 0.0f);
  std::vector<float> total_count_by_words(this->unique_word_num, 0.0f);

  float all_true_count_by_word = 0;
  float all_total_count_by_word = 0;

  auto merged_sorted_index_ = merge_sorted_index_(
    time_data, time_data,
    sorted_index, sorted_index, true
  );
  std::unique_ptr<std::vector<int>> merged_is_src = std::move(merged_sorted_index_.first);
  std::unique_ptr<std::vector<int>> merged_sorted_index = std::move(merged_sorted_index_.second);
  
  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    float target = (*targets)[src_or_dst_index];
    int dst_idx = (*dst_induces)[src_or_dst_index];
    if(is_src) {
      float true_count_of_row = 0;
      float total_count_of_row = 0;
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++
        ) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          true_count_of_row += word_count * (true_count_by_words[word_id] + k1 * all_true_count_by_word / all_total_count_by_word);
          total_count_of_row += word_count * (total_count_by_words[word_id] + k1);
        }
      }
      if(total_count_of_row > 0 || k2 > 0) {
        (*ret)[src_or_dst_index] = (true_count_of_row + k2 * all_true_mean_by_row) / (total_count_of_row + k2);
      } else {
        (*ret)[src_or_dst_index] = all_true_mean_by_row;
      }
    } else {
      if(!std::isnan(target)){
        if(dst_idx >= 0 and dst_idx < this->row_num){
          for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
              word_id_it != this->multi_categorical_values[dst_idx]->end();
              word_id_it++) {
            int word_id = word_id_it->first;
            int word_count = word_id_it->second;
            true_count_by_words[word_id] += target * word_count;
            total_count_by_words[word_id] += word_count;
            all_true_count_by_word += target * word_count;
            all_total_count_by_word += word_count;
          }
        }
      }
    }
  }
  
  return std::move(ret);
}

py::array_t<float> MultiCategoricalManager::temporal_target_encode_with_dst_induces(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  float k1, float k2
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
      std::move(time_data_vec), std::move(sorted_index_vec), k1, k2);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

/*
void MultiCategoricalManager::calculate_hashed_tfidf(
  int dimension
){
  this->hashed_tfidf = std::vector<float>(this->row_num * dimension, 0);
  this->hash_dimension = dimension;
  auto hash_func = std::hash<int>{};

  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    for (auto word_it =  this->tfidf_values[row_idx]->begin();
         word_it != this->tfidf_values[row_idx]->end();
         word_it++) {
        int word_id = word_it->first;
        float tfidf = word_it->second;
        int hash = hash_func(word_id) % dimension;
        this->hashed_tfidf[row_idx * dimension + hash] += tfidf;
     }
  }
}
*/

/*
py::array_t<float> MultiCategoricalManager::get_hashed_tfidf(int idx){
  py::array_t<float> ret = py::array_t<float>({(std::size_t)this->row_num});
  auto ret_r = ret.mutable_unchecked<1>();
  
  if (idx >= this->hash_dimension){
    return ret;
  }
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    ret_r(row_idx) = this->hashed_tfidf[row_idx * this->hash_dimension + idx];
  }
  return ret;
}
*/

std::pair<std::vector<float>, std::vector<float>> MultiCategoricalManager::get_label_count(
  std::vector<float> labels, std::vector<int> dst_induces
) {
  int size = labels.size();
  std::vector<float> true_by_word(this->unique_word_num);
  std::vector<float> count_by_word(this->unique_word_num);
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float label = labels[row_idx];
    int dst_idx = dst_induces[row_idx];
    for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
        word_id_it != this->multi_categorical_values[dst_idx]->end();
        word_id_it++
    ) {
      int word_id = word_id_it->first;
      int word_count = word_id_it->second;

      true_by_word[word_id] += label * word_count;
      count_by_word[word_id] += word_count;
    }
  }
  return {true_by_word, count_by_word};
}


std::unique_ptr<std::vector<bool>> MultiCategoricalManager::calculate_adversarial_valid(
  std::shared_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
  std::shared_ptr<std::vector<int64_t>> adversarial_total_count
) {
  const int size = dst_induces->size();
  std::vector<int64_t> adversarial_true_count_by_word(this->unique_word_num, 0);
  std::vector<int64_t> adversarial_total_count_by_word(this->unique_word_num, 0);
  for(int row_idx = 0; row_idx < size; row_idx++) {
    int dst_idx = (*dst_induces)[row_idx];
    if(dst_idx >= 0 and dst_idx < this->row_num){
      for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
          word_id_it != this->multi_categorical_values[dst_idx]->end();
          word_id_it++
      ) {
        int word_id = word_id_it->first;
        int word_count = word_id_it->second;
        int64_t adversarial_true_count_of_row = (*adversarial_true_count)[row_idx];
        int64_t adversarial_total_count_of_row = (*adversarial_total_count)[row_idx];

        adversarial_true_count_by_word[word_id] += adversarial_true_count_of_row * word_count;
        adversarial_total_count_by_word[word_id] += adversarial_total_count_of_row * word_count;
      }
    }
  }
  
  std::unique_ptr<std::vector<bool>> adverarial_valid
    = std::make_unique<std::vector<bool>>(this->unique_word_num);
  for(int word_id = 0; word_id < this->unique_word_num; word_id++) {
    float ratio
      = (float)adversarial_true_count_by_word[word_id]
      / (float)adversarial_total_count_by_word[word_id];
    if(ratio > 0.99 || ratio < 0.01) {
      (*adverarial_valid)[word_id] = false;
    } else {
      (*adverarial_valid)[word_id] = true;
    }
  }
  
  return std::move(adverarial_valid);
}
std::unique_ptr<std::vector<float>> MultiCategoricalManager::target_encode_with_adversarial_regularization_(
  std::unique_ptr<std::vector<float>> targets,
  std::shared_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
  std::shared_ptr<std::vector<int64_t>> adversarial_total_count,
  float k1, float k2
) {
  std::unique_ptr<std::vector<bool>> adverarial_valid
    = this->calculate_adversarial_valid(dst_induces, adversarial_true_count, adversarial_total_count);

  const int size = targets->size();

  std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);
  
  std::vector<float> true_count_by_words(this->unique_word_num, 0.0f);
  std::vector<float> total_count_by_words(this->unique_word_num, 0.0f);
  
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    if(!std::isnan(target)){
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          if(!(*adverarial_valid)[word_id]){continue;}
          true_count_by_words[word_id] += target * word_count;
          total_count_by_words[word_id] += word_count;
        }
      }
    }
  }

  float all_true_count_by_word = 0;
  float all_total_count_by_word = 0;
  for(int word_id = 0; word_id < this->unique_word_num; word_id++) {
    if(!(*adverarial_valid)[word_id]){continue;}
    all_true_count_by_word += true_count_by_words[word_id];
    all_total_count_by_word += total_count_by_words[word_id];
  }
  float all_true_mean_by_word = all_true_count_by_word / all_total_count_by_word;

  float all_true_count_by_row = 0;
  float all_total_count_by_row = 0;
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    if(!std::isnan(target)){
      all_true_count_by_row += target;
      all_total_count_by_row ++;
    }
  }
  float all_true_mean_by_row = all_true_count_by_row / all_total_count_by_row;

  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    int dst_idx = (*dst_induces)[row_idx];
    float true_count_of_row = 0;
    float total_count_of_row = 0;
    if(!std::isnan(target)){
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++
        ) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          // if(!(*adverarial_valid)[word_id]){continue;}
          true_count_of_row += word_count * (true_count_by_words[word_id] - target * word_count + k1 * all_true_mean_by_word);
          total_count_of_row += word_count * (total_count_by_words[word_id] - word_count + k1);
        }
      }
    } else {
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++
        ) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          // if(!(*adverarial_valid)[word_id]){continue;}
          true_count_of_row += word_count * (true_count_by_words[word_id] + k1 * all_true_mean_by_word);
          total_count_of_row += word_count * (total_count_by_words[word_id] + k1);
        }
      }
    }

    if(total_count_of_row > 0 || k2 > 0) {
      (*ret)[row_idx] = (true_count_of_row + k2 * all_true_mean_by_row) / (total_count_of_row + k2);
    } else {
      (*ret)[row_idx] = all_true_mean_by_row;
    }
  }
  
  return std::move(ret);
}

py::array_t<float> MultiCategoricalManager::target_encode_with_adversarial_regularization(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  py::array_t<int64_t> adversarial_true_count,
  py::array_t<int64_t> adversarial_total_count,
  float k1, float k2
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
      std::move(adversarial_true_count_vec), std::move(adversarial_total_count_vec), k1, k2);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
  
}

std::unique_ptr<std::vector<float>> MultiCategoricalManager::temporal_target_encode_with_adversarial_regularization_(
  std::unique_ptr<std::vector<float>> targets,
  std::shared_ptr<std::vector<int>> dst_induces,
  std::shared_ptr<std::vector<float>> time_data,
  std::shared_ptr<std::vector<int>> sorted_index,
  std::shared_ptr<std::vector<int64_t>> adversarial_true_count,
  std::shared_ptr<std::vector<int64_t>> adversarial_total_count,
  float k1, float k2
){
  std::unique_ptr<std::vector<bool>> adverarial_valid
    = this->calculate_adversarial_valid(dst_induces, adversarial_true_count, adversarial_total_count);

  const int size = targets->size();

  std::unique_ptr<std::vector<float>> ret
    = std::make_unique<std::vector<float>>(size);
  
  float all_true_count_by_row = 0;
  float all_total_count_by_row = 0;
  for(int row_idx = 0; row_idx < size; row_idx++) {
    float target = (*targets)[row_idx];
    if(!std::isnan(target)){
      all_true_count_by_row += target;
      all_total_count_by_row ++;
    }
  }
  float all_true_mean_by_row = all_true_count_by_row / all_total_count_by_row;

  std::vector<float> true_count_by_words(this->unique_word_num, 0.0f);
  std::vector<float> total_count_by_words(this->unique_word_num, 0.0f);

  float all_true_count_by_word = 0;
  float all_total_count_by_word = 0;

  auto merged_sorted_index_ = merge_sorted_index_(
    time_data, time_data,
    sorted_index, sorted_index, true
  );
  std::unique_ptr<std::vector<int>> merged_is_src = std::move(merged_sorted_index_.first);
  std::unique_ptr<std::vector<int>> merged_sorted_index = std::move(merged_sorted_index_.second);
  
  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    float target = (*targets)[src_or_dst_index];
    int dst_idx = (*dst_induces)[src_or_dst_index];
    if(is_src) {
      float true_count_of_row = 0;
      float total_count_of_row = 0;
      if(dst_idx >= 0 and dst_idx < this->row_num){
        for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
            word_id_it != this->multi_categorical_values[dst_idx]->end();
            word_id_it++
        ) {
          int word_id = word_id_it->first;
          int word_count = word_id_it->second;
          // if(!(*adverarial_valid)[word_id]){continue;}
          true_count_of_row += word_count * (true_count_by_words[word_id] + k1 * all_true_count_by_word / all_total_count_by_word);
          total_count_of_row += word_count * (total_count_by_words[word_id] + k1);
        }
      }
      if(total_count_of_row > 0 || k2 > 0) {
        (*ret)[src_or_dst_index] = (true_count_of_row + k2 * all_true_mean_by_row) / (total_count_of_row + k2);
      } else {
        (*ret)[src_or_dst_index] = all_true_mean_by_row;
      }
    } else {
      if(!std::isnan(target)){
        if(dst_idx >= 0 and dst_idx < this->row_num){
          for(auto word_id_it = this->multi_categorical_values[dst_idx]->begin();
              word_id_it != this->multi_categorical_values[dst_idx]->end();
              word_id_it++) {
            int word_id = word_id_it->first;
            int word_count = word_id_it->second;
            if(!(*adverarial_valid)[word_id]){continue;}
            true_count_by_words[word_id] += target * word_count;
            total_count_by_words[word_id] += word_count;
            all_true_count_by_word += target * word_count;
            all_total_count_by_word += word_count;
          }
        }
      }
    }
  }
  
  return std::move(ret);
}

py::array_t<float> MultiCategoricalManager::temporal_target_encode_with_adversarial_regularization(
  py::array_t<float> targets,
  py::array_t<int> dst_induces,
  py::array_t<float> time_data,
  py::array_t<int> sorted_index,
  py::array_t<int64_t> adversarial_true_count,
  py::array_t<int64_t> adversarial_total_count,
  float k1, float k2
) {
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
      k1, k2);
  }
  py::array_t<float> ret = py::array_t<float>(ret_vec->size(), ret_vec->data());
  return ret;
}

/*
py::array_t<float> MultiCategoricalManager::truncated_svd(int rank, bool tfidf, bool log1p){
  py::array_t<float> ret = py::array_t<float>({this->row_num, rank});
  auto ret_r = ret.mutable_unchecked<2>();
  std::vector<Eigen::Triplet<float>> triplet_vec;
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++){
    if(tfidf){
      for(auto tfidf_id_it = this->tfidf_values[row_idx]->begin();
          tfidf_id_it != this->tfidf_values[row_idx]->end();
          tfidf_id_it++) {
        int word_id = tfidf_id_it->first;
        float tfidf = tfidf_id_it->second;
        if(log1p){
          triplet_vec.push_back(Eigen::Triplet<float>(row_idx, word_id, std::log1p(tfidf)));
        }else{
          triplet_vec.push_back(Eigen::Triplet<float>(row_idx, word_id, tfidf));
        }
      }
    }else{
      for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
          word_id_it != this->multi_categorical_values[row_idx]->end();
          word_id_it++) {
        int word_id = word_id_it->first;
        float word_count = word_id_it->second;
        if(log1p){
          triplet_vec.push_back(Eigen::Triplet<float>(row_idx, word_id, std::log1p(word_count)));
        }else{
          triplet_vec.push_back(Eigen::Triplet<float>(row_idx, word_id, word_count));
        }
      }
    }
  }
  
  Eigen::SparseMatrix<float> tfidf_matrix(this->row_num, this->unique_word_num);
  tfidf_matrix.setFromTriplets(triplet_vec.begin(), triplet_vec.end());
  
  RedSVD::RedSVD<Eigen::SparseMatrix<float>> svd(tfidf_matrix, rank);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> U = svd.matrixU();
  
  for(int row_idx = 0; row_idx < this->row_num; row_idx++) {
    for(int rank_idx = 0; rank_idx < rank; rank_idx++) {
      ret_r(row_idx, rank_idx) = U(row_idx, rank_idx);
    }
  }

  return ret;
}
*/

/*
Eigen::SparseMatrix<float> MultiCategoricalManager::get_tfidf_matrix(){
  std::vector<Eigen::Triplet<float>> triplet_vec;

  for(int row_idx = 0; row_idx < this->row_num; row_idx++){
    for(auto tfidf_id_it = this->tfidf_values[row_idx]->begin();
        tfidf_id_it != this->tfidf_values[row_idx]->end();
        tfidf_id_it++) {
      int word_id = tfidf_id_it->first;
      float tfidf = tfidf_id_it->second;
      triplet_vec.push_back(Eigen::Triplet<float>(row_idx, word_id, tfidf));
    }
  }

  Eigen::SparseMatrix<float> tfidf_matrix(this->row_num, this->unique_word_num);
  tfidf_matrix.setFromTriplets(triplet_vec.begin(), triplet_vec.end());
  return tfidf_matrix;
}
*/

Eigen::SparseMatrix<float> MultiCategoricalManager::get_count_matrix(){
  std::vector<Eigen::Triplet<float>> triplet_vec;

  for(int row_idx = 0; row_idx < this->row_num; row_idx++){
    for(auto word_id_it = this->multi_categorical_values[row_idx]->begin();
        word_id_it != this->multi_categorical_values[row_idx]->end();
        word_id_it++) {
      int word_id = word_id_it->first;
      float count = word_id_it->second;
      triplet_vec.push_back(Eigen::Triplet<float>(row_idx, word_id, count));
    }
  }

  Eigen::SparseMatrix<float> count_matrix(this->row_num, this->unique_word_num);
  count_matrix.setFromTriplets(triplet_vec.begin(), triplet_vec.end());
  return count_matrix;
}

void InitMultiCategoricalManager(pybind11::module& m){
  m.doc() = "multi categorical manager made by pybind11"; // optional

  py::class_<MultiCategoricalManager>(m, "MultiCategoricalManager")
    .def(py::init<std::vector<std::string>>())
    .def_readonly("total_word_num", &MultiCategoricalManager::total_word_num)
    .def_readonly("max_word_num", &MultiCategoricalManager::max_word_num)
    .def_readonly("mean_word_num", &MultiCategoricalManager::mean_word_num)
    .def_readonly("row_num", &MultiCategoricalManager::row_num)
    .def_readonly("unique_word_num", &MultiCategoricalManager::unique_word_num)
    .def_readonly("words_sorted_by_num", &MultiCategoricalManager::words_sorted_by_num)
    // .def_readonly("tfidf_of_1word", &MultiCategoricalManager::tfidf_of_1word)
    .def("target_encode", &MultiCategoricalManager::target_encode)
    // .def("tfidf_target_encode", &MultiCategoricalManager::tfidf_target_encode)
    .def("temporal_target_encode", &MultiCategoricalManager::temporal_target_encode)
    .def("target_encode_with_dst_induces",
         &MultiCategoricalManager::target_encode_with_dst_induces)
    .def("temporal_target_encode_with_dst_induces",
         &MultiCategoricalManager::temporal_target_encode_with_dst_induces)
    .def("length", &MultiCategoricalManager::length)
    .def("nunique", &MultiCategoricalManager::nunique)
    .def("duplicates", &MultiCategoricalManager::duplicates)
    .def("mode", &MultiCategoricalManager::mode)
    // .def("max_tfidf_words", &MultiCategoricalManager::max_tfidf_words)
    .def("max_count", &MultiCategoricalManager::max_count)
    .def("min_count", &MultiCategoricalManager::min_count)
    // .def("tfidf", &MultiCategoricalManager::tfidf)
    .def("count", &MultiCategoricalManager::count)
    // .def("calculate_hashed_tfidf", &MultiCategoricalManager::calculate_hashed_tfidf)
    // .def("get_hashed_tfidf", &MultiCategoricalManager::get_hashed_tfidf)
    .def("get_label_count", &MultiCategoricalManager::get_label_count)
    .def("target_encode_with_adversarial_regularization",
         &MultiCategoricalManager::target_encode_with_adversarial_regularization)
    .def("temporal_target_encode_with_adversarial_regularization",
         &MultiCategoricalManager::temporal_target_encode_with_adversarial_regularization)
    // .def("truncated_svd",
    //      &MultiCategoricalManager::truncated_svd)
    // .def("get_tfidf_matrix",
    //      &MultiCategoricalManager::get_tfidf_matrix)
    .def("get_count_matrix",
         &MultiCategoricalManager::get_count_matrix);
}
