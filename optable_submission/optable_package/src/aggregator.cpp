#include "aggregator.h"


std::pair<std::unique_ptr<std::vector<float>>, std::unique_ptr<std::vector<int>>> get_src_time_and_sorted_index_(
  std::shared_ptr<std::vector<float>> dst_time,
  std::shared_ptr<std::vector<int>> src_id,
  std::shared_ptr<std::vector<int>> dst_id
) {
  unsigned int dst_size = dst_id->size();
  unsigned int src_size = src_id->size();

  std::unique_ptr<std::vector<float>> src_time
    = std::make_unique<std::vector<float>>(src_size);
  std::unique_ptr<std::vector<int>> src_sorted_index
    = std::make_unique<std::vector<int>>(src_size);
  for(int src_idx = 0; src_idx < src_size; src_idx++) {
    (*src_sorted_index)[src_idx] = src_idx;
  }

  std::unordered_map<int, float> time_by_id = std::unordered_map<int, float>{};
  
  for(int dst_idx = 0; dst_idx < dst_size; dst_idx++) {
    if((*dst_id)[dst_idx] >= 0) {
      time_by_id[(*dst_id)[dst_idx]] = (*dst_time)[dst_idx];
    }
  }
  
  for(int src_idx = 0; src_idx < src_size; src_idx++) {
    if(time_by_id.find((*src_id)[src_idx]) != time_by_id.end()){
      (*src_time)[src_idx] = time_by_id[(*src_id)[src_idx]];
    } else {
      (*src_time)[src_idx] = std::nanf("");
    }
  }


  std::sort(src_sorted_index->begin(),
            src_sorted_index->end(),
            [&src_time](size_t i1, size_t i2) {
              if(std::isnan((*src_time)[i1])){
                return false;
              } else if(std::isnan((*src_time)[i2])) {
                return true;
              } else {
                return (*src_time)[i1] < (*src_time)[i2];
              }
            });

  return {std::move(src_time), std::move(src_sorted_index)};
}

std::pair<py::array_t<float>, py::array_t<int>> get_src_time_and_sorted_index(
  py::array_t<float> dst_time,
  py::array_t<int> src_id,
  py::array_t<int> dst_id
) {
  const int src_size = src_id.size();
  const int dst_size = dst_id.size();

  py::buffer_info dst_time_buf = dst_time.request();
  float *dst_time_ptr = (float *) dst_time_buf.ptr;

  py::buffer_info src_id_buf = src_id.request();
  int *src_id_ptr = (int *) src_id_buf.ptr;

  py::buffer_info dst_id_buf = dst_id.request();
  int *dst_id_ptr = (int *) dst_id_buf.ptr;
  
  std::pair<std::unique_ptr<std::vector<float>>, std::unique_ptr<std::vector<int>>> ret_vecs;
  {
    py::gil_scoped_release release;
    std::shared_ptr<std::vector<float>> dst_time_vec
      = std::make_shared<std::vector<float>>(dst_time_ptr, dst_time_ptr + dst_size);
    std::shared_ptr<std::vector<int>> src_id_vec
      = std::make_shared<std::vector<int>>(src_id_ptr, src_id_ptr + src_size);
    std::shared_ptr<std::vector<int>> dst_id_vec
      = std::make_shared<std::vector<int>>(dst_id_ptr, dst_id_ptr + dst_size);
    
    ret_vecs = get_src_time_and_sorted_index_(dst_time_vec, src_id_vec, dst_id_vec);
  }
  
  std::pair<py::array_t<float>, py::array_t<int>> ret = {
    py::array_t<float>(ret_vecs.first->size(), ret_vecs.first->data()),
    py::array_t<int>(ret_vecs.second->size(), ret_vecs.second->data())
  };

  return ret;
}

std::pair<std::unique_ptr<std::vector<int>>, std::unique_ptr<std::vector<int>>> merge_sorted_index_(
  std::shared_ptr<std::vector<float>> src_time_data,
  std::shared_ptr<std::vector<float>> dst_time_data,
  std::shared_ptr<std::vector<int>> src_sorted_index,
  std::shared_ptr<std::vector<int>> dst_sorted_index,
  bool is_src_priority
)
{
  int src_size = src_time_data->size();
  int dst_size = dst_time_data->size();
  std::unique_ptr<std::vector<int>> merged_is_src
    = std::make_unique<std::vector<int>>(src_size + dst_size);
  std::unique_ptr<std::vector<int>> merged_sorted_index
    = std::make_unique<std::vector<int>>(src_size + dst_size);
  int src_index_of_sorted_index = 0;
  int dst_index_of_sorted_index = 0;
  
  int src_tmp_index = (*src_sorted_index)[src_index_of_sorted_index];
  int dst_tmp_index = (*dst_sorted_index)[dst_index_of_sorted_index];
  float src_tmp_time = (*src_time_data)[src_tmp_index];
  float dst_tmp_time = (*dst_time_data)[dst_tmp_index];
  int merged_idx = 0;
  while(src_index_of_sorted_index < src_size || dst_index_of_sorted_index < dst_size) {
    if (src_index_of_sorted_index >= src_size) {
      (*merged_is_src)[merged_idx] = false;
      (*merged_sorted_index)[merged_idx] = dst_tmp_index;

      dst_index_of_sorted_index++;
      if(dst_index_of_sorted_index < dst_size){
        dst_tmp_index = (*dst_sorted_index)[dst_index_of_sorted_index];
        dst_tmp_time = (*dst_time_data)[dst_tmp_index];
      }
    } else if (dst_index_of_sorted_index >= dst_size) {
      (*merged_is_src)[merged_idx] = true;
      (*merged_sorted_index)[merged_idx] = src_tmp_index;

      src_index_of_sorted_index++;
      if(src_index_of_sorted_index < src_size){
        src_tmp_index = (*src_sorted_index)[src_index_of_sorted_index];
        src_tmp_time = (*src_time_data)[src_tmp_index];
      }
    } else if (std::isnan(src_tmp_time) || src_tmp_time > dst_tmp_time || ((!is_src_priority) && src_tmp_time == dst_tmp_time)) {
      (*merged_is_src)[merged_idx] = false;
      (*merged_sorted_index)[merged_idx] = dst_tmp_index;

      dst_index_of_sorted_index++;
      if(dst_index_of_sorted_index < dst_size){
        dst_tmp_index = (*dst_sorted_index)[dst_index_of_sorted_index];
        dst_tmp_time = (*dst_time_data)[dst_tmp_index];
      }
    } else {
      (*merged_is_src)[merged_idx] = true;
      (*merged_sorted_index)[merged_idx] = src_tmp_index;

      src_index_of_sorted_index++;
      if(src_index_of_sorted_index < src_size){
        src_tmp_index = (*src_sorted_index)[src_index_of_sorted_index];
        src_tmp_time = (*src_time_data)[src_tmp_index];
      }
    }
    merged_idx++;
  }
  
  return {std::move(merged_is_src), std::move(merged_sorted_index)};  
}

std::pair<py::array_t<int>, py::array_t<int>> merge_sorted_index(
  py::array_t<float> src_time_data,
  py::array_t<float> dst_time_data,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index,
  bool is_src_priority = true
) {
  const int src_size = src_time_data.size();
  const int dst_size = dst_time_data.size();
  py::buffer_info src_time_data_buf = src_time_data.request();
  float *src_time_data_ptr = (float *) src_time_data_buf.ptr;

  py::buffer_info dst_time_data_buf = dst_time_data.request();
  float *dst_time_data_ptr = (float *) dst_time_data_buf.ptr;
  
  py::buffer_info src_sorted_index_buf = src_sorted_index.request();
  int *src_sorted_index_ptr = (int *) src_sorted_index_buf.ptr;

  py::buffer_info dst_sorted_index_buf = dst_sorted_index.request();
  int *dst_sorted_index_ptr = (int *) dst_sorted_index_buf.ptr;

  std::pair<std::unique_ptr<std::vector<int>>, std::unique_ptr<std::vector<int>>> ret_vecs;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> src_time_data_vec
      = std::make_unique<std::vector<float>>(src_time_data_ptr, src_time_data_ptr + src_size);
    std::unique_ptr<std::vector<float>> dst_time_data_vec
      = std::make_unique<std::vector<float>>(dst_time_data_ptr, dst_time_data_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_sorted_index_vec
      = std::make_unique<std::vector<int>>(src_sorted_index_ptr, src_sorted_index_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_sorted_index_vec
      = std::make_unique<std::vector<int>>(dst_sorted_index_ptr, dst_sorted_index_ptr + dst_size);
    
    ret_vecs = merge_sorted_index_(
      std::move(src_time_data_vec), std::move(dst_time_data_vec),
      std::move(src_sorted_index_vec), std::move(dst_sorted_index_vec), is_src_priority);
  }
  
  std::pair<py::array_t<int>, py::array_t<int>> ret = {
    py::array_t<int>(ret_vecs.first->size(), ret_vecs.first->data()),
    py::array_t<int>(ret_vecs.second->size(), ret_vecs.second->data())
  };

  return ret;
}

std::shared_ptr<std::vector<int>> squash_splited_relations_(
  std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations,
  std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations
) {
  int relations_num = src_id_for_each_relations.size();
  int tables_num = relations_num + 1;
  
  std::shared_ptr<std::vector<int>> dst_id_for_current_table;
  std::shared_ptr<std::vector<int>> dst_id_for_previous_table;
  dst_id_for_previous_table = src_id_for_each_relations[relations_num - 1];
  for(int table_idx = relations_num - 2; table_idx >= 0; table_idx--) {
    std::shared_ptr<std::vector<int>> current_id_for_src = src_id_for_each_relations[table_idx];
    std::shared_ptr<std::vector<int>> current_id_for_dst = dst_id_for_each_relations[table_idx];
    dst_id_for_current_table = std::make_unique<std::vector<int>>(current_id_for_src->size());

    std::unordered_map<int, int> dst_id_by_current_id = std::unordered_map<int, int>{};
    for(int current_dst_index = 0; current_dst_index < current_id_for_dst->size(); current_dst_index++) {
      int current_id_for_dst_idx = (*current_id_for_dst)[current_dst_index];
      if(current_id_for_dst_idx >= 0) {
        dst_id_by_current_id[current_id_for_dst_idx]
          = (*dst_id_for_previous_table)[current_dst_index];
      }
    }
    for(int current_src_index = 0; current_src_index < current_id_for_src->size(); current_src_index++) {
      int current_id_for_src_idx = (*current_id_for_src)[current_src_index];
      if(dst_id_by_current_id.find(current_id_for_src_idx) != dst_id_by_current_id.end()){
        (*dst_id_for_current_table)[current_src_index] = dst_id_by_current_id[current_id_for_src_idx];
      } else {
        (*dst_id_for_current_table)[current_src_index] = -1;
      }
    }

    dst_id_for_previous_table = dst_id_for_current_table;
  }
  
  return dst_id_for_previous_table;
};

py::array_t<int> squash_splited_relations(
  std::vector<py::array_t<int>> src_id_for_each_relations,
  std::vector<py::array_t<int>> dst_id_for_each_relations
) {
  const int relations_num = src_id_for_each_relations.size();
  std::vector<int> src_id_size = {};
  std::vector<int> dst_id_size = {};
  std::vector<int*> src_id_ptr = {};
  std::vector<int*> dst_id_ptr = {};

  for (int rel_idx = 0; rel_idx < relations_num; rel_idx++) {
    py::buffer_info src_id_buf = src_id_for_each_relations[rel_idx].request();
    src_id_ptr.push_back((int *)src_id_buf.ptr);
    src_id_size.push_back(src_id_for_each_relations[rel_idx].size());
    
    py::buffer_info dst_id_buf = dst_id_for_each_relations[rel_idx].request();
    dst_id_ptr.push_back((int *)dst_id_buf.ptr);
    dst_id_size.push_back(dst_id_for_each_relations[rel_idx].size());
  }
  
  std::shared_ptr<std::vector<int>> ret_vec;
  {
    py::gil_scoped_release release;
    std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations_vecs = {};
    std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations_vecs = {};
    for (int rel_idx = 0; rel_idx < relations_num; rel_idx++) {
      std::shared_ptr<std::vector<int>> src_tmp_vec
        = std::make_shared<std::vector<int>>(src_id_ptr[rel_idx], src_id_ptr[rel_idx] + src_id_size[rel_idx]);
      src_id_for_each_relations_vecs.push_back(src_tmp_vec);
      std::shared_ptr<std::vector<int>> dst_tmp_vec
        = std::make_shared<std::vector<int>>(dst_id_ptr[rel_idx], dst_id_ptr[rel_idx] + dst_id_size[rel_idx]);
      dst_id_for_each_relations_vecs.push_back(dst_tmp_vec);
    }
      
    ret_vec = squash_splited_relations_(
      src_id_for_each_relations_vecs,
      dst_id_for_each_relations_vecs);
  }
  
  return py::array_t<int>(ret_vec->size(), ret_vec->data());
};

std::unique_ptr<std::vector<float>> temporal_to_one_aggregate_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::shared_ptr<std::vector<int>> src_ids,
  std::shared_ptr<std::vector<int>> dst_ids,
  std::shared_ptr<std::vector<float>> src_time,
  std::shared_ptr<std::vector<float>> dst_time,
  std::shared_ptr<std::vector<int>> src_sorted_index,
  std::shared_ptr<std::vector<int>> dst_sorted_index
) {
  const int dst_size = dst_ids->size();
  const int src_size = src_ids->size();

  auto merged_sorted_index_ = merge_sorted_index_(
    std::move(src_time), std::move(dst_time),
    std::move(src_sorted_index), std::move(dst_sorted_index), false
  );
  std::unique_ptr<std::vector<int>> merged_is_src
    = std::move(merged_sorted_index_.first);
  std::unique_ptr<std::vector<int>> merged_sorted_index
    = std::move(merged_sorted_index_.second);

  std::unique_ptr<std::vector<float>> src_data
    = std::make_unique<std::vector<float>>(src_size);
  
  std::unordered_map<int, float> data_by_id = std::unordered_map<int, float>{};

  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    if(is_src) {
      int src_id = (*src_ids)[src_or_dst_index];
      if (data_by_id.find(src_id) != data_by_id.end()) {
        (*src_data)[src_or_dst_index] = data_by_id[src_id];
      } else {
        (*src_data)[src_or_dst_index] = std::nanf("");
      }
    } else {
      int dst_id = (*dst_ids)[src_or_dst_index];
      if (dst_id >= 0){
        data_by_id[dst_id] = (*dst_data)[src_or_dst_index];
      }
    }
  }

  return std::move(src_data);
}

py::array_t<float> temporal_to_one_aggregate(
  py::array_t<float> dst_data,
  py::array_t<int> src_ids,
  py::array_t<int> dst_ids,
  py::array_t<float> src_time,
  py::array_t<float> dst_time,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index
) {
  const int dst_size = dst_ids.size();
  const int src_size = src_ids.size();

  py::buffer_info dst_data_buf = dst_data.request();
  float *dst_data_ptr = (float *) dst_data_buf.ptr;
  
  py::buffer_info src_ids_buf = src_ids.request();
  int *src_ids_ptr = (int *) src_ids_buf.ptr;
  
  py::buffer_info dst_ids_buf = dst_ids.request();
  int *dst_ids_ptr = (int *) dst_ids_buf.ptr;

  py::buffer_info src_time_buf = src_time.request();
  float *src_time_ptr = (float *) src_time_buf.ptr;

  py::buffer_info dst_time_buf = dst_time.request();
  float *dst_time_ptr = (float *) dst_time_buf.ptr;

  py::buffer_info src_sorted_index_buf = src_sorted_index.request();
  int *src_sorted_index_ptr = (int *) src_sorted_index_buf.ptr;

  py::buffer_info dst_sorted_index_buf = dst_sorted_index.request();
  int *dst_sorted_index_ptr = (int *) dst_sorted_index_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> dst_data_vec
      = std::make_unique<std::vector<float>>(dst_data_ptr, dst_data_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_ids_vec
      = std::make_unique<std::vector<int>>(src_ids_ptr, src_ids_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_ids_vec
      = std::make_unique<std::vector<int>>(dst_ids_ptr, dst_ids_ptr + dst_size);
    std::unique_ptr<std::vector<float>> src_time_vec
      = std::make_unique<std::vector<float>>(src_time_ptr, src_time_ptr + src_size);
    std::unique_ptr<std::vector<float>> dst_time_vec
      = std::make_unique<std::vector<float>>(dst_time_ptr, dst_time_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_sorted_index_vec
      = std::make_unique<std::vector<int>>(src_sorted_index_ptr, src_sorted_index_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_sorted_index_vec
      = std::make_unique<std::vector<int>>(dst_sorted_index_ptr, dst_sorted_index_ptr + dst_size);
  
    ret_vec = temporal_to_one_aggregate_(
      std::move(dst_data_vec), std::move(src_ids_vec), std::move(dst_ids_vec), std::move(src_time_vec), std::move(dst_time_vec),
      std::move(src_sorted_index_vec), std::move(dst_sorted_index_vec));
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}

std::unique_ptr<std::vector<float>> not_temporal_to_one_aggregate_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::shared_ptr<std::vector<int>> src_ids,
  std::shared_ptr<std::vector<int>> dst_ids
) {
  const int dst_size = dst_ids->size();
  const int src_size = src_ids->size();
  
  std::unique_ptr<std::vector<float>> src_data
    = std::make_unique<std::vector<float>>(src_size);
  std::unordered_map<int, float> data_by_id = std::unordered_map<int, float>{};
  
  for(int dst_idx = 0; dst_idx < dst_size; dst_idx++) {
    int dst_id = (*dst_ids)[dst_idx];
    if (dst_id >= 0) {
      data_by_id[dst_id] = (*dst_data)[dst_idx];
    }
  }
  
  for(int src_idx = 0; src_idx < src_size; src_idx++) {
    int src_id = (*src_ids)[src_idx];
    if (data_by_id.find(src_id) != data_by_id.end()) {
      (*src_data)[src_idx] = data_by_id[src_id];
    } else {
      (*src_data)[src_idx] = std::nanf("");
    }
  }
  
  return std::move(src_data);
}

py::array_t<float> not_temporal_to_one_aggregate(
  py::array_t<float> dst_data,
  py::array_t<int> src_ids,
  py::array_t<int> dst_ids
) {
  const int dst_size = dst_ids.size();
  const int src_size = src_ids.size();

  py::buffer_info dst_data_buf = dst_data.request();
  float *dst_data_ptr = (float *) dst_data_buf.ptr;
  
  py::buffer_info src_ids_buf = src_ids.request();
  int *src_ids_ptr = (int *) src_ids_buf.ptr;
  
  py::buffer_info dst_ids_buf = dst_ids.request();
  int *dst_ids_ptr = (int *) dst_ids_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> dst_data_vec
      = std::make_unique<std::vector<float>>(dst_data_ptr, dst_data_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_ids_vec
      = std::make_unique<std::vector<int>>(src_ids_ptr, src_ids_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_ids_vec
      = std::make_unique<std::vector<int>>(dst_ids_ptr, dst_ids_ptr + dst_size);
  
    ret_vec = not_temporal_to_one_aggregate_(
      std::move(dst_data_vec), std::move(src_ids_vec), std::move(dst_ids_vec));
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}


std::unique_ptr<std::vector<float>> time_splited_to_one_aggregate_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations,
  std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations,
  std::shared_ptr<std::vector<float>> src_time,
  std::shared_ptr<std::vector<float>> dst_time,
  std::shared_ptr<std::vector<int>> src_sorted_index,
  std::shared_ptr<std::vector<int>> dst_sorted_index
) {
  int relations_num = src_id_for_each_relations.size();
  int tables_num = relations_num + 1;
  std::shared_ptr<std::vector<int>> squashed_id = squash_splited_relations_(
    src_id_for_each_relations,
    dst_id_for_each_relations);
  std::unique_ptr<std::vector<float>> ret = temporal_to_one_aggregate_(
    std::move(dst_data),
    squashed_id,
    dst_id_for_each_relations[relations_num-1],
    src_time,
    dst_time,
    src_sorted_index,
    dst_sorted_index);
  return std::move(ret);
}

py::array_t<float> time_splited_to_one_aggregate(
  py::array_t<float> dst_data,
  std::vector<py::array_t<int>> src_id_for_each_relations,
  std::vector<py::array_t<int>> dst_id_for_each_relations,
  py::array_t<float> src_time,
  py::array_t<float> dst_time,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index
) {
  const int src_size = src_time.size();
  const int dst_size = dst_time.size();
  const int relations_num = src_id_for_each_relations.size();
  std::vector<int> src_id_size = {};
  std::vector<int> dst_id_size = {};
  std::vector<int*> src_id_ptr = {};
  std::vector<int*> dst_id_ptr = {};

  for (int rel_idx = 0; rel_idx < relations_num; rel_idx++) {
    py::buffer_info src_id_buf = src_id_for_each_relations[rel_idx].request();
    src_id_ptr.push_back((int *)src_id_buf.ptr);
    src_id_size.push_back(src_id_for_each_relations[rel_idx].size());
    
    py::buffer_info dst_id_buf = dst_id_for_each_relations[rel_idx].request();
    dst_id_ptr.push_back((int *)dst_id_buf.ptr);
    dst_id_size.push_back(dst_id_for_each_relations[rel_idx].size());
  }

  py::buffer_info dst_data_buf = dst_data.request();
  float *dst_data_ptr = (float *) dst_data_buf.ptr;
  
  py::buffer_info src_time_buf = src_time.request();
  float *src_time_ptr = (float *) src_time_buf.ptr;

  py::buffer_info dst_time_buf = dst_time.request();
  float *dst_time_ptr = (float *) dst_time_buf.ptr;
  
  py::buffer_info src_sorted_index_buf = src_sorted_index.request();
  int *src_sorted_index_ptr = (int *) src_sorted_index_buf.ptr;
  
  py::buffer_info dst_sorted_index_buf = dst_sorted_index.request();
  int *dst_sorted_index_ptr = (int *) dst_sorted_index_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations_vecs = {};
    std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations_vecs = {};
    for (int rel_idx = 0; rel_idx < relations_num; rel_idx++) {
      std::shared_ptr<std::vector<int>> src_tmp_vec
        = std::make_shared<std::vector<int>>(src_id_ptr[rel_idx], src_id_ptr[rel_idx] + src_id_size[rel_idx]);
      src_id_for_each_relations_vecs.push_back(src_tmp_vec);
      std::shared_ptr<std::vector<int>> dst_tmp_vec
        = std::make_shared<std::vector<int>>(dst_id_ptr[rel_idx], dst_id_ptr[rel_idx] + dst_id_size[rel_idx]);
      dst_id_for_each_relations_vecs.push_back(dst_tmp_vec);
    }
    std::unique_ptr<std::vector<float>> dst_data_vec
      = std::make_unique<std::vector<float>>(dst_data_ptr, dst_data_ptr + dst_size);
    std::unique_ptr<std::vector<float>> src_time_vec
      = std::make_unique<std::vector<float>>(src_time_ptr, src_time_ptr + src_size);
    std::unique_ptr<std::vector<float>> dst_time_vec
      = std::make_unique<std::vector<float>>(dst_time_ptr, dst_time_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_sorted_index_vec
      = std::make_unique<std::vector<int>>(src_sorted_index_ptr, src_sorted_index_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_sorted_index_vec
      = std::make_unique<std::vector<int>>(dst_sorted_index_ptr, dst_sorted_index_ptr + dst_size);
      
    ret_vec = time_splited_to_one_aggregate_(
      std::move(dst_data_vec),
      src_id_for_each_relations_vecs,
      dst_id_for_each_relations_vecs,
      std::move(src_time_vec),
      std::move(dst_time_vec),
      std::move(src_sorted_index_vec),
      std::move(dst_sorted_index_vec)
    );
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}

std::unique_ptr<std::vector<float>> temporal_to_many_aggregate_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::shared_ptr<std::vector<int>> src_ids,
  std::shared_ptr<std::vector<int>> dst_ids,
  std::shared_ptr<std::vector<float>> src_time,
  std::shared_ptr<std::vector<float>> dst_time,
  std::shared_ptr<std::vector<int>> src_sorted_index,
  std::shared_ptr<std::vector<int>> dst_sorted_index,
  std::string mode
) {
  unsigned int dst_size = dst_ids->size();
  unsigned int src_size = src_ids->size();

  auto merged_sorted_index_ = merge_sorted_index_(
    std::move(src_time), std::move(dst_time),
    std::move(src_sorted_index), std::move(dst_sorted_index), false
  );
  std::unique_ptr<std::vector<int>> merged_is_src
    = std::move(merged_sorted_index_.first);
  std::unique_ptr<std::vector<int>> merged_sorted_index
    = std::move(merged_sorted_index_.second);

  std::unique_ptr<std::vector<float>> src_data
    = std::make_unique<std::vector<float>>(src_size);
  
  std::unordered_map<int, std::unique_ptr<AggregateSet>> agg_set_by_id = std::unordered_map<int, std::unique_ptr<AggregateSet>>{};
  std::unordered_map<int, std::unordered_map<float, int>> value_num_map_by_id = std::unordered_map<int, std::unordered_map<float, int>>{};
  std::unordered_map<int, int> nunique_by_id = std::unordered_map<int, int>{};
  std::unordered_map<int, float> mode_by_id = std::unordered_map<int, float>{};
  std::unordered_map<int, int> mode_freq_by_id = std::unordered_map<int, int>{};
  std::unordered_map<int, int> duplicates_by_id = std::unordered_map<int, int>{};

  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    if(is_src) {
      int id = (*src_ids)[src_or_dst_index];
      float data;
      if (mode == "nunique") {
        if(nunique_by_id.find(id) == nunique_by_id.end()) {
          data = 0.0;
        } else {
          data = (float)nunique_by_id[id];
        }
      } else if (mode == "mode") {
        if(mode_by_id.find(id) == mode_by_id.end()) {
          data = std::nanf("");
        } else {
          data = mode_by_id[id];
        }
      } else if (mode == "duplicates") {
        if(duplicates_by_id.find(id) == duplicates_by_id.end()) {
          data = std::nanf("");
        } else {
          data = (float)duplicates_by_id[id];
        }
      } else {
        if(agg_set_by_id.find(id) == agg_set_by_id.end()){
          if (mode == "nunique" || mode == "variance" || mode == "sum") {
            data = 0.0;
          } else {
            data = std::nanf("");
          }
        } else {
          data = agg_set_by_id[id]->get();
        }
      }
      (*src_data)[src_or_dst_index] = data;
    } else {
      int id = (*dst_ids)[src_or_dst_index];
      if (id >= 0) {
        float data = (*dst_data)[src_or_dst_index];
        if(std::isnan(data)){
          continue;
        }
        if(agg_set_by_id.find(id) == agg_set_by_id.end()){
          if (mode == "sum") {
            agg_set_by_id[id] = std::make_unique<SumAggregateSet>();
          } else if (mode == "variance"){
            agg_set_by_id[id] = std::make_unique<VarianceAggregateSet>();
          } else if (mode == "max") {
            agg_set_by_id[id] = std::make_unique<MaxAggregateSet>();
          } else if (mode == "min") {
            agg_set_by_id[id] = std::make_unique<MinAggregateSet>();
          } else if (mode == "mean") {
            agg_set_by_id[id] = std::make_unique<MeanAggregateSet>();
          } else if (mode == "median") {
            agg_set_by_id[id] = std::make_unique<MedianAggregateSet>();
          } else if (mode == "nunique") {
            // agg_set_by_id[id] = std::make_unique<NUniqueAggregateSet>();
          } else if (mode == "last"){
            agg_set_by_id[id] = std::make_unique<LastAggregateSet>();
          } else if (mode == "mode"){
            // agg_set_by_id[id] = std::make_unique<ModeAggregateSet>();
          } else if (mode == "mode_ratio"){
            agg_set_by_id[id] = std::make_unique<ModeRatioAggregateSet>();
          } else if (mode == "duplicates") {
            // agg_set_by_id[id] = std::make_unique<DuplicatesAggregateSet>();
          } else if (mode == "kurtosis") {
            agg_set_by_id[id] = std::make_unique<KurtosisAggregateSet>();
          } else if (mode == "skewness") {
            agg_set_by_id[id] = std::make_unique<SkewnessAggregateSet>();
          } else if (mode == "rolling_sum10") {
            agg_set_by_id[id] = std::make_unique<RollingSum10AggregateSet>();
          } else if (mode == "rolling_mean10") {
            agg_set_by_id[id] = std::make_unique<RollingMean10AggregateSet>();
          } else if (mode == "delay10") {
            agg_set_by_id[id] = std::make_unique<Delay10AggregateSet>();
          } else {
            throw std::runtime_error("aggregate mode is invalid. (temporal_to_many_aggregate_)");
          }
        }
        if (mode == "nunique") {
          if(value_num_map_by_id.find(id) == value_num_map_by_id.end()){
            value_num_map_by_id[id] = std::unordered_map<float, int>{};
            nunique_by_id[id] = 0;
          }
          if(value_num_map_by_id[id].find(data) == value_num_map_by_id[id].end()){
            value_num_map_by_id[id][data] = 1;
            nunique_by_id[id]++;
          } else {
            (++value_num_map_by_id[id][data]);
          }
        } else if (mode == "mode") {
          if(value_num_map_by_id.find(id) == value_num_map_by_id.end()){
            value_num_map_by_id[id] = std::unordered_map<float, int>{};
            mode_freq_by_id[id] = 0;
          }
          int current_freq;
          if(value_num_map_by_id[id].find(data) == value_num_map_by_id[id].end()){
            current_freq = (value_num_map_by_id[id][data] = 1);
          } else {
            current_freq = (++value_num_map_by_id[id][data]);
          }
          if (current_freq >= mode_freq_by_id[id]) {
            mode_by_id[id] = data;
            mode_freq_by_id[id] = current_freq;
          }
        } else if (mode == "duplicates") {
          if(value_num_map_by_id.find(id) == value_num_map_by_id.end()){
            value_num_map_by_id[id] = std::unordered_map<float, int>{};
            duplicates_by_id[id] = 0;
          }
          if(value_num_map_by_id[id].find(data) == value_num_map_by_id[id].end()){
            value_num_map_by_id[id][data] = 1;
          } else {
            duplicates_by_id[id]++;
            (++value_num_map_by_id[id][data]);
          }
        } else {
          agg_set_by_id[id]->set_value(data);
        }
      }
    }
  }
  return std::move(src_data);
}

py::array_t<float> temporal_to_many_aggregate(
  py::array_t<float> dst_data,
  py::array_t<int> src_id,
  py::array_t<int> dst_id,
  py::array_t<float> src_time,
  py::array_t<float> dst_time,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index,
  std::string mode
) {
  const int src_size = src_time.size();
  const int dst_size = dst_time.size();  

  py::buffer_info dst_data_buf = dst_data.request();
  float *dst_data_ptr = (float *) dst_data_buf.ptr;

  py::buffer_info src_id_buf = src_id.request();
  int *src_id_ptr = (int *) src_id_buf.ptr;

  py::buffer_info dst_id_buf = dst_id.request();
  int *dst_id_ptr = (int *) dst_id_buf.ptr;

  py::buffer_info src_time_buf = src_time.request();
  float *src_time_ptr = (float *) src_time_buf.ptr;

  py::buffer_info dst_time_buf = dst_time.request();
  float *dst_time_ptr = (float *) dst_time_buf.ptr;

  py::buffer_info src_sorted_index_buf = src_sorted_index.request();
  int *src_sorted_index_ptr = (int *) src_sorted_index_buf.ptr;

  py::buffer_info dst_sorted_index_buf = dst_sorted_index.request();
  int *dst_sorted_index_ptr = (int *) dst_sorted_index_buf.ptr;

  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> dst_data_vec
      = std::make_unique<std::vector<float>>(dst_data_ptr, dst_data_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_id_vec
      = std::make_unique<std::vector<int>>(src_id_ptr, src_id_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_id_vec
      = std::make_unique<std::vector<int>>(dst_id_ptr, dst_id_ptr + dst_size);
    std::unique_ptr<std::vector<float>> src_time_vec
      = std::make_unique<std::vector<float>>(src_time_ptr, src_time_ptr + src_size);
    std::unique_ptr<std::vector<float>> dst_time_vec
      = std::make_unique<std::vector<float>>(dst_time_ptr, dst_time_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_sorted_index_vec
      = std::make_unique<std::vector<int>>(src_sorted_index_ptr, src_sorted_index_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_sorted_index_vec
      = std::make_unique<std::vector<int>>(dst_sorted_index_ptr, dst_sorted_index_ptr + dst_size);
    
    ret_vec = temporal_to_many_aggregate_(
      std::move(dst_data_vec), std::move(src_id_vec), std::move(dst_id_vec),
      std::move(src_time_vec), std::move(dst_time_vec),
      std::move(src_sorted_index_vec), std::move(dst_sorted_index_vec), mode
    );
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}


std::unique_ptr<std::vector<float>> not_temporal_to_many_aggregate_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::shared_ptr<std::vector<int>> src_ids,
  std::shared_ptr<std::vector<int>> dst_ids,
  std::string mode
) {
  unsigned int dst_size = dst_ids->size();
  unsigned int src_size = src_ids->size();

  std::unique_ptr<std::vector<float>> src_data
    = std::make_unique<std::vector<float>>(src_size);
  
  std::unordered_map<int, std::unique_ptr<AggregateSet>>agg_set_by_id = std::unordered_map<int, std::unique_ptr<AggregateSet>>{};
  std::unordered_map<int, std::unordered_map<float, int>> value_num_map_by_id = std::unordered_map<int, std::unordered_map<float, int>>{};
  std::unordered_map<int, int> nunique_by_id = std::unordered_map<int, int>{};
  std::unordered_map<int, float> mode_by_id = std::unordered_map<int, float>{};
  std::unordered_map<int, int> mode_freq_by_id = std::unordered_map<int, int>{};
  std::unordered_map<int, int> duplicates_by_id = std::unordered_map<int, int>{};

  for(int dst_idx = 0; dst_idx < dst_size; dst_idx++) {
    int id = (*dst_ids)[dst_idx];
    if (id >= 0){
      float data = (*dst_data)[dst_idx];
      if(std::isnan(data)){
        continue;
      }
      if(agg_set_by_id.find(id) == agg_set_by_id.end()){
        if (mode == "sum") {
          agg_set_by_id[id] = std::make_unique<SumAggregateSet>();
        } else if (mode == "variance"){
          agg_set_by_id[id] = std::make_unique<VarianceAggregateSet>();
        } else if (mode == "max") {
          agg_set_by_id[id] = std::make_unique<MaxAggregateSet>();
        } else if (mode == "min") {
          agg_set_by_id[id] = std::make_unique<MinAggregateSet>();
        } else if (mode == "mean") {
          agg_set_by_id[id] = std::make_unique<MeanAggregateSet>();
        } else if (mode == "median") {
          agg_set_by_id[id] = std::make_unique<MedianAggregateSet>();
        } else if (mode == "nunique") {
          // agg_set_by_id[id] = std::make_unique<NUniqueAggregateSet>();
        } else if (mode == "last"){
          agg_set_by_id[id] = std::make_unique<LastAggregateSet>();
        } else if (mode == "mode"){
          // agg_set_by_id[id] = std::make_unique<ModeAggregateSet>();
        } else if (mode == "mode_ratio"){
          agg_set_by_id[id] = std::make_unique<ModeRatioAggregateSet>();
        } else if (mode == "duplicates"){
          // agg_set_by_id[id] = std::make_unique<DuplicatesAggregateSet>();
        } else if (mode == "kurtosis") {
          agg_set_by_id[id] = std::make_unique<KurtosisAggregateSet>();
        } else if (mode == "skewness") {
          agg_set_by_id[id] = std::make_unique<SkewnessAggregateSet>();
        } else if (mode == "rolling_sum10") {
          agg_set_by_id[id] = std::make_unique<RollingSum10AggregateSet>();
        } else if (mode == "rolling_mean10") {
          agg_set_by_id[id] = std::make_unique<RollingMean10AggregateSet>();
        } else if (mode == "delay10") {
          agg_set_by_id[id] = std::make_unique<Delay10AggregateSet>();
        } else {
          throw std::runtime_error("aggregate mode is invalid. (temporal_to_many_aggregate_)");
        }
      }
      if (mode == "nunique") {
        if(value_num_map_by_id.find(id) == value_num_map_by_id.end()){
          value_num_map_by_id[id] = std::unordered_map<float, int>{};
          nunique_by_id[id] = 0;
        }
        if(value_num_map_by_id[id].find(data) == value_num_map_by_id[id].end()){
          value_num_map_by_id[id][data] = 1;
          nunique_by_id[id]++;
        } else {
          (++value_num_map_by_id[id][data]);
        }
      } else if (mode == "mode") {
        if(value_num_map_by_id.find(id) == value_num_map_by_id.end()){
          value_num_map_by_id[id] = std::unordered_map<float, int>{};
          mode_freq_by_id[id] = 0;
        }
        int current_freq;
        if(value_num_map_by_id[id].find(data) == value_num_map_by_id[id].end()){
          current_freq = (value_num_map_by_id[id][data] = 1);
        } else {
          current_freq = (++value_num_map_by_id[id][data]);
        }
        if (current_freq >= mode_freq_by_id[id]) {
          mode_by_id[id] = data;
          mode_freq_by_id[id] = current_freq;
        }
      } else if (mode == "duplicates") {
        if(value_num_map_by_id.find(id) == value_num_map_by_id.end()){
          value_num_map_by_id[id] = std::unordered_map<float, int>{};
          duplicates_by_id[id] = 0;
        }
        if(value_num_map_by_id[id].find(data) == value_num_map_by_id[id].end()){
          value_num_map_by_id[id][data] = 1;
        } else {
          duplicates_by_id[id]++;
          (++value_num_map_by_id[id][data]);
        }
      } else {
        agg_set_by_id[id]->set_value(data);
      }
    }
  }
  for(int src_idx = 0; src_idx < src_size; src_idx++) {
    int id = (*src_ids)[src_idx];
    float data;
    if (mode == "nunique") {
      if(nunique_by_id.find(id) == nunique_by_id.end()) {
        data = 0.0;
      } else {
        data = (float)nunique_by_id[id];
      }
    } else if (mode == "mode") {
      if(mode_by_id.find(id) == mode_by_id.end()) {
        data = std::nanf("");
      } else {
        data = mode_by_id[id];
      }
    } else if (mode == "duplicates") {
      if(duplicates_by_id.find(id) == duplicates_by_id.end()) {
        data = std::nanf("");
      } else {
        data = (float)duplicates_by_id[id];
      }
    } else {
      if(agg_set_by_id.find(id) == agg_set_by_id.end()){
        if (mode == "nunique" || mode == "variance" || mode == "sum") {
          data = 0.0;
        } else {
          data = std::nanf("");
        }
      } else {
        data = agg_set_by_id[id]->get();
      }
    }
    (*src_data)[src_idx] = data;
  }
  
  return std::move(src_data);
}

py::array_t<float> not_temporal_to_many_aggregate(
  py::array_t<float> dst_data,
  py::array_t<int> src_id,
  py::array_t<int> dst_id,
  std::string mode
) {
  const int src_size = src_id.size();
  const int dst_size = dst_id.size();  

  py::buffer_info dst_data_buf = dst_data.request();
  float *dst_data_ptr = (float *) dst_data_buf.ptr;

  py::buffer_info src_id_buf = src_id.request();
  int *src_id_ptr = (int *) src_id_buf.ptr;

  py::buffer_info dst_id_buf = dst_id.request();
  int *dst_id_ptr = (int *) dst_id_buf.ptr;

  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::unique_ptr<std::vector<float>> dst_data_vec
      = std::make_unique<std::vector<float>>(dst_data_ptr, dst_data_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_id_vec
      = std::make_unique<std::vector<int>>(src_id_ptr, src_id_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_id_vec
      = std::make_unique<std::vector<int>>(dst_id_ptr, dst_id_ptr + dst_size);
    
    ret_vec = not_temporal_to_many_aggregate_(
      std::move(dst_data_vec), std::move(src_id_vec), std::move(dst_id_vec), mode
    );
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}


std::unique_ptr<std::vector<float>> time_splited_to_many_aggregate_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations,
  std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations,
  std::shared_ptr<std::vector<float>> src_time,
  std::shared_ptr<std::vector<float>> dst_time,
  std::shared_ptr<std::vector<int>> src_sorted_index,
  std::shared_ptr<std::vector<int>> dst_sorted_index,
  std::string mode
) {
  int relations_num = src_id_for_each_relations.size();
  int tables_num = relations_num + 1;
  std::shared_ptr<std::vector<int>> squashed_id = squash_splited_relations_(src_id_for_each_relations, dst_id_for_each_relations);
  std::unique_ptr<std::vector<float>> ret =  temporal_to_many_aggregate_(
    std::move(dst_data),
    squashed_id,
    dst_id_for_each_relations[relations_num-1],
    src_time,
    dst_time,
    src_sorted_index,
    dst_sorted_index,
    mode);
  return std::move(ret);
}

py::array_t<float> time_splited_to_many_aggregate(
  py::array_t<float> dst_data,
  std::vector<py::array_t<int>> src_id_for_each_relations,
  std::vector<py::array_t<int>> dst_id_for_each_relations,
  py::array_t<float> src_time,
  py::array_t<float> dst_time,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index,
  std::string mode
) {
  const int src_size = src_time.size();
  const int dst_size = dst_time.size();
  const int relations_num = src_id_for_each_relations.size();
  std::vector<int> src_id_size = {};
  std::vector<int> dst_id_size = {};
  std::vector<int*> src_id_ptr = {};
  std::vector<int*> dst_id_ptr = {};

  for (int rel_idx = 0; rel_idx < relations_num; rel_idx++) {
    py::buffer_info src_id_buf = src_id_for_each_relations[rel_idx].request();
    src_id_ptr.push_back((int *)src_id_buf.ptr);
    src_id_size.push_back(src_id_for_each_relations[rel_idx].size());
    
    py::buffer_info dst_id_buf = dst_id_for_each_relations[rel_idx].request();
    dst_id_ptr.push_back((int *)dst_id_buf.ptr);
    dst_id_size.push_back(dst_id_for_each_relations[rel_idx].size());
  }

  py::buffer_info dst_data_buf = dst_data.request();
  float *dst_data_ptr = (float *) dst_data_buf.ptr;
  
  py::buffer_info src_time_buf = src_time.request();
  float *src_time_ptr = (float *) src_time_buf.ptr;

  py::buffer_info dst_time_buf = dst_time.request();
  float *dst_time_ptr = (float *) dst_time_buf.ptr;
  
  py::buffer_info src_sorted_index_buf = src_sorted_index.request();
  int *src_sorted_index_ptr = (int *) src_sorted_index_buf.ptr;
  
  py::buffer_info dst_sorted_index_buf = dst_sorted_index.request();
  int *dst_sorted_index_ptr = (int *) dst_sorted_index_buf.ptr;
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations_vecs = {};
    std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations_vecs = {};
    for (int rel_idx = 0; rel_idx < relations_num; rel_idx++) {
      std::shared_ptr<std::vector<int>> src_tmp_vec
        = std::make_unique<std::vector<int>>(src_id_ptr[rel_idx], src_id_ptr[rel_idx] + src_id_size[rel_idx]);
      src_id_for_each_relations_vecs.push_back(std::move(src_tmp_vec));
      std::shared_ptr<std::vector<int>> dst_tmp_vec
        = std::make_unique<std::vector<int>>(dst_id_ptr[rel_idx], dst_id_ptr[rel_idx] + dst_id_size[rel_idx]);
      dst_id_for_each_relations_vecs.push_back(std::move(dst_tmp_vec));
    }
    std::unique_ptr<std::vector<float>> dst_data_vec
      = std::make_unique<std::vector<float>>(dst_data_ptr, dst_data_ptr + dst_size);
    std::unique_ptr<std::vector<float>> src_time_vec
      = std::make_unique<std::vector<float>>(src_time_ptr, src_time_ptr + src_size);
    std::unique_ptr<std::vector<float>> dst_time_vec
      = std::make_unique<std::vector<float>>(dst_time_ptr, dst_time_ptr + dst_size);
    std::unique_ptr<std::vector<int>> src_sorted_index_vec
      = std::make_unique<std::vector<int>>(src_sorted_index_ptr, src_sorted_index_ptr + src_size);
    std::unique_ptr<std::vector<int>> dst_sorted_index_vec
      = std::make_unique<std::vector<int>>(dst_sorted_index_ptr, dst_sorted_index_ptr + dst_size);
      
    ret_vec = time_splited_to_many_aggregate_(
      std::move(dst_data_vec),
      src_id_for_each_relations_vecs,
      dst_id_for_each_relations_vecs,
      std::move(src_time_vec),
      std::move(dst_time_vec),
      std::move(src_sorted_index_vec),
      std::move(dst_sorted_index_vec),
      mode
    );
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}

std::unique_ptr<std::vector<float>> Aggregator::aggregate_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::unordered_map<int, std::shared_ptr<std::vector<float>>> time_for_each_table,
  std::unordered_map<int, std::shared_ptr<std::vector<int>>> sorted_index_for_each_table,
  std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relation,
  std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relation,
  std::vector<bool> src_is_unique_for_each_relation,
  std::vector<bool> dst_is_unique_for_each_relation,
  std::string mode1,
  std::string mode2) {
  // TODO: table_idxとtable_idx +1がどこだかこんがらがってしまうので考える
  int relations_num = src_id_for_each_relation.size();
  int tables_num = relations_num + 1;

  /*
  for(int table_idx = relations_num - 1; table_idx >= 0; table_idx--) {
    if(time_for_each_table.find(table_idx) == time_for_each_table.end()
       && time_for_each_table.find(table_idx + 1) != time_for_each_table.end()
       && dst_is_unique_for_each_relation[table_idx]) {
         auto src_time_and_sorted_index = get_src_time_and_sorted_index_(
           time_for_each_table[table_idx + 1],
           src_id_for_each_relation[table_idx],
           dst_id_for_each_relation[table_idx]
         );
         time_for_each_table[table_idx] = std::move(src_time_and_sorted_index.first);
         sorted_index_for_each_table[table_idx] = std::move(src_time_and_sorted_index.second);
     }
  }
  */

  bool once_to_many_done = false;
  std::unique_ptr<std::vector<float>> current_dst_data = std::move(dst_data);
  int table_idx = relations_num - 1;
  while(table_idx >= 0) {
    if (time_for_each_table.find(table_idx+1) == time_for_each_table.end()) {
      if(dst_is_unique_for_each_relation[table_idx]){
        current_dst_data = not_temporal_to_one_aggregate_(
          std::move(current_dst_data),
          src_id_for_each_relation[table_idx],
          dst_id_for_each_relation[table_idx]);
      } else{
        if(once_to_many_done) {
          current_dst_data = not_temporal_to_many_aggregate_(
            std::move(current_dst_data),
            src_id_for_each_relation[table_idx],
            dst_id_for_each_relation[table_idx],
            mode2);
        } else {
          current_dst_data = not_temporal_to_many_aggregate_(
            std::move(current_dst_data),
            src_id_for_each_relation[table_idx],
            dst_id_for_each_relation[table_idx],
            mode1);
          once_to_many_done = true;
        }
      }
      table_idx--;
    } else{
      if (time_for_each_table.find(table_idx) == time_for_each_table.end()) {
        bool needed_multiple_aggregate = false;
        int search_table_idx = table_idx - 1;
        while(search_table_idx >= 0) {
          if (!dst_is_unique_for_each_relation[search_table_idx]){
            break;
          }
          if (time_for_each_table.find(search_table_idx) != time_for_each_table.end()) {
            needed_multiple_aggregate = true;
            break;
          }
          search_table_idx--;
        }
        
        if(needed_multiple_aggregate){
          if(dst_is_unique_for_each_relation[table_idx]){
            std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations_for_aggregate = std::vector<std::shared_ptr<std::vector<int>>>{};
            std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations_for_aggregate = std::vector<std::shared_ptr<std::vector<int>>>{};
            for(int tmp_table_idx = search_table_idx; tmp_table_idx < table_idx + 1; tmp_table_idx++) {
              src_id_for_each_relations_for_aggregate.push_back(src_id_for_each_relation[tmp_table_idx]);
              dst_id_for_each_relations_for_aggregate.push_back(dst_id_for_each_relation[tmp_table_idx]);
            }
            current_dst_data = time_splited_to_one_aggregate_(
              std::move(current_dst_data),
              src_id_for_each_relations_for_aggregate,
              dst_id_for_each_relations_for_aggregate,
              time_for_each_table[search_table_idx],
              time_for_each_table[table_idx+1],
              sorted_index_for_each_table[search_table_idx],
              sorted_index_for_each_table[table_idx+1]
            );
          } else {
            std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relations_for_aggregate = std::vector<std::shared_ptr<std::vector<int>>>{};
            std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relations_for_aggregate = std::vector<std::shared_ptr<std::vector<int>>>{};
            for(int tmp_table_idx = search_table_idx; tmp_table_idx < table_idx + 1; tmp_table_idx++) {
              src_id_for_each_relations_for_aggregate.push_back(src_id_for_each_relation[tmp_table_idx]);
              dst_id_for_each_relations_for_aggregate.push_back(dst_id_for_each_relation[tmp_table_idx]);
            }
            if(once_to_many_done) {
              current_dst_data = time_splited_to_many_aggregate_(
                std::move(current_dst_data),
                src_id_for_each_relations_for_aggregate,
                dst_id_for_each_relations_for_aggregate,
                time_for_each_table[search_table_idx],
                time_for_each_table[table_idx+1],
                sorted_index_for_each_table[search_table_idx],
                sorted_index_for_each_table[table_idx+1],
                mode2
              );
            } else {
              current_dst_data = time_splited_to_many_aggregate_(
                std::move(current_dst_data),
                src_id_for_each_relations_for_aggregate,
                dst_id_for_each_relations_for_aggregate,
                time_for_each_table[search_table_idx],
                time_for_each_table[table_idx+1],
                sorted_index_for_each_table[search_table_idx],
                sorted_index_for_each_table[table_idx+1],
                mode1
              );
              once_to_many_done = true;
            }
          }
          table_idx = search_table_idx-1;
        } else{
          if(dst_is_unique_for_each_relation[table_idx]){
            current_dst_data = not_temporal_to_one_aggregate_(
              std::move(current_dst_data),
              src_id_for_each_relation[table_idx],
              dst_id_for_each_relation[table_idx]);
          }else{
            if(once_to_many_done) {
              current_dst_data = not_temporal_to_many_aggregate_(
                std::move(current_dst_data),
                src_id_for_each_relation[table_idx],
                dst_id_for_each_relation[table_idx],
                mode2);
            } else {
              current_dst_data = not_temporal_to_many_aggregate_(
                std::move(current_dst_data),
                src_id_for_each_relation[table_idx],
                dst_id_for_each_relation[table_idx],
                mode1);
              once_to_many_done = true;
            }
          }
          table_idx--;
        }
      } else {
        if(dst_is_unique_for_each_relation[table_idx]){
          current_dst_data = temporal_to_one_aggregate_(
            std::move(current_dst_data),
            src_id_for_each_relation[table_idx],
            dst_id_for_each_relation[table_idx],
            time_for_each_table[table_idx],
            time_for_each_table[table_idx+1],
            sorted_index_for_each_table[table_idx],
            sorted_index_for_each_table[table_idx+1]);
        }else{
          if(once_to_many_done) {
            current_dst_data = temporal_to_many_aggregate_(
              std::move(current_dst_data),
              src_id_for_each_relation[table_idx],
              dst_id_for_each_relation[table_idx],
              time_for_each_table[table_idx],
              time_for_each_table[table_idx+1],
              sorted_index_for_each_table[table_idx],
              sorted_index_for_each_table[table_idx+1],
              mode2);
          } else {
            current_dst_data = temporal_to_many_aggregate_(
              std::move(current_dst_data),
              src_id_for_each_relation[table_idx],
              dst_id_for_each_relation[table_idx],
              time_for_each_table[table_idx],
              time_for_each_table[table_idx+1],
              sorted_index_for_each_table[table_idx],
              sorted_index_for_each_table[table_idx+1],
              mode1);
            once_to_many_done = true;
          }
        }
        table_idx--;
      }
    }
  }
  return std::move(current_dst_data);
}

py::array_t<float> Aggregator::aggregate(
  py::array_t<float> dst_data,
  std::unordered_map<int, py::array_t<float>> time_for_each_table,
  std::unordered_map<int, py::array_t<int>> sorted_index_for_each_table,
  std::vector<py::array_t<int>> src_id_for_each_relation,
  std::vector<py::array_t<int>> dst_id_for_each_relation,
  std::vector<bool> src_is_unique_for_each_relation,
  std::vector<bool> dst_is_unique_for_each_relation,
  std::string mode1,
  std::string mode2
) {
  int table_num = src_id_for_each_relation.size() + 1;
  std::vector<int> size_of_each_table(table_num);
  for(int table_idx = 0; table_idx < table_num; table_idx++) {
    if(table_idx == table_num - 1) {
      size_of_each_table[table_idx] = dst_id_for_each_relation[table_idx-1].size();
    } else {
      size_of_each_table[table_idx] = src_id_for_each_relation[table_idx].size();
    }
  }
  
  py::buffer_info dst_data_buf = dst_data.request();
  float *dst_data_ptr = (float *) dst_data_buf.ptr;

  std::unordered_map<int, float*> time_for_each_table_ptr = std::unordered_map<int, float*>{};
  std::unordered_map<int, int*> sorted_index_for_each_table_ptr = std::unordered_map<int, int*>{};
  std::vector<int*> src_id_for_each_relation_ptr = {};
  std::vector<int*> dst_id_for_each_relation_ptr = {};
  
  for(auto time_for_each_table_it = time_for_each_table.begin();
      time_for_each_table_it != time_for_each_table.end();
      time_for_each_table_it++) {
    int table_idx = time_for_each_table_it->first;
    py::array_t<float> time_array = time_for_each_table_it->second;
    py::buffer_info time_buf = time_array.request();
    time_for_each_table_ptr[table_idx] = (float *) time_buf.ptr;
  }
  
  for(auto sorted_index_for_each_table_it = sorted_index_for_each_table.begin();
      sorted_index_for_each_table_it != sorted_index_for_each_table.end();
      sorted_index_for_each_table_it++) {
    int table_idx = sorted_index_for_each_table_it->first;
    py::array_t<int> sorted_index_array = sorted_index_for_each_table_it->second;
    py::buffer_info sorted_index_buf = sorted_index_array.request();
    sorted_index_for_each_table_ptr[table_idx] = (int *) sorted_index_buf.ptr;
  }
  
  for(int table_idx = 0; table_idx < table_num-1; table_idx++){
    py::buffer_info src_id_buf = src_id_for_each_relation[table_idx].request();
    src_id_for_each_relation_ptr.push_back((int*) src_id_buf.ptr);
    py::buffer_info dst_id_buf = dst_id_for_each_relation[table_idx].request();
    dst_id_for_each_relation_ptr.push_back((int*) dst_id_buf.ptr);
  }
  
  std::unique_ptr<std::vector<float>> ret_vec;
  {
    py::gil_scoped_release release;
    
    std::unique_ptr<std::vector<float>> dst_data_vec
      = std::make_unique<std::vector<float>>(dst_data_ptr, dst_data_ptr + size_of_each_table[table_num - 1]);
      
    std::unordered_map<int, std::shared_ptr<std::vector<float>>> time_for_each_table_vecs
      = std::unordered_map<int, std::shared_ptr<std::vector<float>>>{};
    for(auto time_for_each_table_it = time_for_each_table_ptr.begin();
        time_for_each_table_it != time_for_each_table_ptr.end();
        time_for_each_table_it++) {
      int table_idx = time_for_each_table_it->first;
      float * time_ptr = time_for_each_table_it->second;
      time_for_each_table_vecs[table_idx] = std::make_unique<std::vector<float>>(
        time_ptr, time_ptr + size_of_each_table[table_idx]
      );
    }
    
    std::unordered_map<int, std::shared_ptr<std::vector<int>>> sorted_index_for_each_table_vecs
      = std::unordered_map<int, std::shared_ptr<std::vector<int>>>{};
    for(auto sorted_index_for_each_table_it = sorted_index_for_each_table_ptr.begin();
        sorted_index_for_each_table_it != sorted_index_for_each_table_ptr.end();
        sorted_index_for_each_table_it++) {
      int table_idx = sorted_index_for_each_table_it->first;
      int *sorted_index_ptr = sorted_index_for_each_table_it->second;
      sorted_index_for_each_table_vecs[table_idx] = std::make_unique<std::vector<int>>(
        sorted_index_ptr, sorted_index_ptr + size_of_each_table[table_idx]
      );
    }

    std::vector<std::shared_ptr<std::vector<int>>> src_id_for_each_relation_vecs = {};
    std::vector<std::shared_ptr<std::vector<int>>> dst_id_for_each_relation_vecs = {};
    for(int table_idx = 0; table_idx < table_num-1; table_idx++){
      int *src_id_ptr = src_id_for_each_relation_ptr[table_idx];
      int src_size = size_of_each_table[table_idx];
      src_id_for_each_relation_vecs.push_back(
        std::make_unique<std::vector<int>>(
          src_id_ptr, src_id_ptr + src_size
        )
      );
      
      int *dst_id_ptr = dst_id_for_each_relation_ptr[table_idx];
      int dst_size = size_of_each_table[table_idx + 1];
      dst_id_for_each_relation_vecs.push_back(
        std::make_unique<std::vector<int>>(
          dst_id_ptr, dst_id_ptr + dst_size
        )
      );
    }
    
    ret_vec = this->aggregate_(
      std::move(dst_data_vec),
      time_for_each_table_vecs,
      sorted_index_for_each_table_vecs,
      src_id_for_each_relation_vecs,
      dst_id_for_each_relation_vecs,
      src_is_unique_for_each_relation,
      dst_is_unique_for_each_relation,
      mode1,
      mode2);
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}

void InitAggregator(pybind11::module& m){
  m.doc() = "aggregator made by pybind11";

  py::class_<Aggregator>(m, "Aggregator")
    .def(py::init<>())
    .def("aggregate", &Aggregator::aggregate);

  m.def("merge_sorted_index", &merge_sorted_index);
  m.def("squash_splited_relations", &squash_splited_relations);
  m.def("get_src_time_and_sorted_index", &get_src_time_and_sorted_index);
  m.def("temporal_to_one_aggregate", &temporal_to_one_aggregate);
  m.def("not_temporal_to_one_aggregate", &not_temporal_to_one_aggregate);
  m.def("time_splited_to_one_aggregate", &time_splited_to_one_aggregate);
  m.def("not_temporal_to_many_aggregate", &not_temporal_to_many_aggregate);
  m.def("temporal_to_many_aggregate", &temporal_to_many_aggregate);
  m.def("time_splited_to_many_aggregate", &time_splited_to_many_aggregate);
}
