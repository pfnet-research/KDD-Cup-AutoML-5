#include "sequential_regression.h"


std::unique_ptr<std::vector<float>> sequential_regression_aggregation_(
  std::unique_ptr<std::vector<float>> dst_data,
  std::shared_ptr<std::vector<int>> src_ids,
  std::shared_ptr<std::vector<int>> dst_ids,
  std::shared_ptr<std::vector<float>> src_time,
  std::shared_ptr<std::vector<float>> dst_time,
  std::shared_ptr<std::vector<int>> src_sorted_index,
  std::shared_ptr<std::vector<int>> dst_sorted_index
) {
  unsigned int dst_size = dst_ids->size();
  unsigned int src_size = src_ids->size();

  auto merged_sorted_index_ = merge_sorted_index_(
    src_time, dst_time,
    std::move(src_sorted_index), std::move(dst_sorted_index), false
  );
  std::unique_ptr<std::vector<int>> merged_is_src
    = std::move(merged_sorted_index_.first);
  std::unique_ptr<std::vector<int>> merged_sorted_index
    = std::move(merged_sorted_index_.second);

  std::unique_ptr<std::vector<float>> coefficient_ret
    = std::make_unique<std::vector<float>>(src_size);
  
  std::unordered_map<int, MEAN_ACC_SET> x_mean_set = std::unordered_map<int, MEAN_ACC_SET>{};
  std::unordered_map<int, MEAN_ACC_SET> t_mean_set = std::unordered_map<int, MEAN_ACC_SET>{};
  std::unordered_map<int, MEAN_ACC_SET> xt_mean_set = std::unordered_map<int, MEAN_ACC_SET>{};
  std::unordered_map<int, MEAN_ACC_SET> xx_mean_set = std::unordered_map<int, MEAN_ACC_SET>{};
  
  for(int merged_idx = 0; merged_idx < merged_is_src->size(); merged_idx++){
    int is_src = (*merged_is_src)[merged_idx];
    int src_or_dst_index = (*merged_sorted_index)[merged_idx];
    if(is_src) {
      int id = (*src_ids)[src_or_dst_index];
      float coefficient;
      float t_data = (*src_time)[src_or_dst_index];
      if (x_mean_set.find(id) == x_mean_set.end()){
        coefficient = std::nanf("");
      } else {
        // TODO: src_timeでintercept
        float x_mean = boost::accumulators::extract::mean(x_mean_set[id]);
        float t_mean = boost::accumulators::extract::mean(t_mean_set[id]);
        float xt_mean = boost::accumulators::extract::mean(xt_mean_set[id]);
        float xx_mean = boost::accumulators::extract::mean(xx_mean_set[id]);
        coefficient = (xt_mean - (x_mean * t_mean)) / (xx_mean - (x_mean * x_mean));
      }
      (*coefficient_ret)[src_or_dst_index] = coefficient;
    } else {
      int id = (*dst_ids)[src_or_dst_index];
      if (id >= 0) {
        float x_data = (*dst_data)[src_or_dst_index];
        float t_data = (*dst_time)[src_or_dst_index];
        if(std::isnan(x_data)){
          continue;
        }
        if(x_mean_set.find(id) == x_mean_set.end()){
          x_mean_set[id] = MEAN_ACC_SET{};
          t_mean_set[id] = MEAN_ACC_SET{};
          xt_mean_set[id] = MEAN_ACC_SET{};
          xx_mean_set[id] = MEAN_ACC_SET{};
        }
        x_mean_set[id](x_data);
        t_mean_set[id](t_data);
        xt_mean_set[id](x_data * t_data);
        xx_mean_set[id](x_data * x_data);
      }
    }
  }

  return std::move(coefficient_ret);
}

py::array_t<float> sequential_regression_aggregation(
  py::array_t<float> dst_data,
  py::array_t<int> src_id,
  py::array_t<int> dst_id,
  py::array_t<float> src_time,
  py::array_t<float> dst_time,
  py::array_t<int> src_sorted_index,
  py::array_t<int> dst_sorted_index
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
    // py::gil_scoped_release release;
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
    
    ret_vec = sequential_regression_aggregation_(
      std::move(dst_data_vec), std::move(src_id_vec), std::move(dst_id_vec),
      std::move(src_time_vec), std::move(dst_time_vec),
      std::move(src_sorted_index_vec), std::move(dst_sorted_index_vec)
    );
  }
  
  return py::array_t<float>(ret_vec->size(), ret_vec->data());
}


void InitSequentialRegression(pybind11::module& m){
  m.def("sequential_regression_aggregation", &sequential_regression_aggregation);
}