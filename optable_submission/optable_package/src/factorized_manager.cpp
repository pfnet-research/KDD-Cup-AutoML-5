#include "factorized_manager.h"



FactorizedManager::FactorizedManager(
  std::vector<CategoricalManager *> categorical_managers_,
  std::vector<MultiCategoricalManager *> multi_categorical_managers_
) {
  this->categorical_managers = categorical_managers_;
  this->multi_categorical_managers = multi_categorical_managers_;
}


Eigen::SparseMatrix<float> FactorizedManager::get_factorized_matrix(){
  int total_id_num = 0;
  int row_num;
  for(auto c_mngr_it = this->categorical_managers.begin();
      c_mngr_it != this->categorical_managers.end();
      c_mngr_it++){
    total_id_num += (*c_mngr_it)->unique_num;
    row_num = (*c_mngr_it)->row_num;
  }
  for(auto m_mngr_it = this->multi_categorical_managers.begin();
      m_mngr_it != this->multi_categorical_managers.end();
      m_mngr_it++) {
    total_id_num += (*m_mngr_it)->unique_word_num;
    row_num = (*m_mngr_it)->row_num;
  }
  
  std::vector<Eigen::Triplet<float>> triplet_vec;
  int current_base_id = 0;
  for(auto c_mngr_it = this->categorical_managers.begin();
      c_mngr_it != this->categorical_managers.end();
      c_mngr_it++){

    for(int row_idx = 0; row_idx < row_num; row_idx++){
      int cat_id = (*c_mngr_it)->categorical_values[row_idx];
      if(cat_id >= 0){
        triplet_vec.push_back(Eigen::Triplet<float>(row_idx, current_base_id+cat_id, 1));
      }
    }
    current_base_id += (*c_mngr_it)->unique_num;
  }
  for(auto m_mngr_it = this->multi_categorical_managers.begin();
      m_mngr_it != this->multi_categorical_managers.end();
      m_mngr_it++) {
    for(int row_idx = 0; row_idx < row_num; row_idx++){
      for(auto multi_cat_id_it = (*m_mngr_it)->multi_categorical_values[row_idx]->begin();
          multi_cat_id_it != (*m_mngr_it)->multi_categorical_values[row_idx]->end();
          multi_cat_id_it++) {
        int word_id = multi_cat_id_it->first;
        triplet_vec.push_back(Eigen::Triplet<float>(row_idx, current_base_id+word_id, 1));
      }
    }
    current_base_id += (*m_mngr_it)->unique_word_num;
  }
  
  Eigen::SparseMatrix<float> factorized_matrix(row_num, total_id_num);
  factorized_matrix.setFromTriplets(triplet_vec.begin(), triplet_vec.end());
  return factorized_matrix;
}

py::array_t<float> FactorizedManager::truncated_svd(int rank){
  int total_id_num = 0;
  int row_num;
  for(auto c_mngr_it = this->categorical_managers.begin();
      c_mngr_it != this->categorical_managers.end();
      c_mngr_it++){
    total_id_num += (*c_mngr_it)->unique_num;
    row_num = (*c_mngr_it)->row_num;
  }
  for(auto m_mngr_it = this->multi_categorical_managers.begin();
      m_mngr_it != this->multi_categorical_managers.end();
      m_mngr_it++) {
    total_id_num += (*m_mngr_it)->unique_word_num;
    row_num = (*m_mngr_it)->row_num;
  }
  
  std::vector<Eigen::Triplet<float>> triplet_vec;
  int current_base_id = 0;
  for(auto c_mngr_it = this->categorical_managers.begin();
      c_mngr_it != this->categorical_managers.end();
      c_mngr_it++){

    for(int row_idx = 0; row_idx < row_num; row_idx++){
      int cat_id = (*c_mngr_it)->categorical_values[row_idx];
      if(cat_id >= 0){
        triplet_vec.push_back(Eigen::Triplet<float>(row_idx, current_base_id+cat_id, 1));
      }
    }
    current_base_id += (*c_mngr_it)->unique_num;
  }
  for(auto m_mngr_it = this->multi_categorical_managers.begin();
      m_mngr_it != this->multi_categorical_managers.end();
      m_mngr_it++) {
    for(int row_idx = 0; row_idx < row_num; row_idx++){
      for(auto multi_cat_id_it = (*m_mngr_it)->multi_categorical_values[row_idx]->begin();
          multi_cat_id_it != (*m_mngr_it)->multi_categorical_values[row_idx]->end();
          multi_cat_id_it++) {
        int word_id = multi_cat_id_it->first;
        triplet_vec.push_back(Eigen::Triplet<float>(row_idx, current_base_id+word_id, 1));
      }
    }
    current_base_id += (*m_mngr_it)->unique_word_num;
  }
  
  Eigen::SparseMatrix<float> factorized_matrix(row_num, total_id_num);
  factorized_matrix.setFromTriplets(triplet_vec.begin(), triplet_vec.end());
  
  RedSVD::RedSVD<Eigen::SparseMatrix<float>> svd(factorized_matrix, rank);
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> U = svd.matrixU();
  
  py::array_t<float> ret = py::array_t<float>({row_num, rank});
  auto ret_r = ret.mutable_unchecked<2>();
  
  for(int row_idx = 0; row_idx < row_num; row_idx++) {
    for(int rank_idx = 0; rank_idx < rank; rank_idx++) {
      ret_r(row_idx, rank_idx) = U(row_idx, rank_idx);
    }
  }

  return ret;
}


void InitFactorizedManager(pybind11::module& m){
  m.doc() = "factorized manager made by pybind11"; // optional

  py::class_<FactorizedManager>(m, "FactorizedManager")
    .def(py::init<std::vector<CategoricalManager *>, std::vector<MultiCategoricalManager *>>())
    .def("get_factorized_matrix", &FactorizedManager::get_factorized_matrix)
    .def("truncated_svd", &FactorizedManager::truncated_svd);
}