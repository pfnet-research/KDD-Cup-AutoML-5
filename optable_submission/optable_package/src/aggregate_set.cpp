#include "aggregate_set.h"

SumAggregateSet::SumAggregateSet(){
  this->sum_acc_set = SUM_ACC_SET{};
}

void SumAggregateSet::set_value(float value){
  this->sum_acc_set(value);
}

float SumAggregateSet::get() {
  return boost::accumulators::extract::sum(this->sum_acc_set);
}

VarianceAggregateSet::VarianceAggregateSet(){
  this->variance_acc_set = VARIANCE_ACC_SET{};
}

void VarianceAggregateSet::set_value(float value){
  this->variance_acc_set(value);
}

float VarianceAggregateSet::get() {
  return boost::accumulators::extract::variance(this->variance_acc_set);
}

MaxAggregateSet::MaxAggregateSet(){
  this->max_acc_set = MAX_ACC_SET{};
}

void MaxAggregateSet::set_value(float value) {
  this->max_acc_set(value);
}

float MaxAggregateSet::get() {
  return boost::accumulators::extract::max(this->max_acc_set);
}

MinAggregateSet::MinAggregateSet(){
  this->min_acc_set = MIN_ACC_SET{};
}

void MinAggregateSet::set_value(float value) {
  this->min_acc_set(value);
}

float MinAggregateSet::get() {
  return boost::accumulators::extract::min(this->min_acc_set);
}

MeanAggregateSet::MeanAggregateSet(){
  this->mean_acc_set = MEAN_ACC_SET{};
}

void MeanAggregateSet::set_value(float value) {
  this->mean_acc_set(value);
}

float MeanAggregateSet::get() {
  return boost::accumulators::extract::mean(this->mean_acc_set);
}

MedianAggregateSet::MedianAggregateSet(){
  this->median_acc_set = MEDIAN_ACC_SET{};
}

void MedianAggregateSet::set_value(float value) {
  this->median_acc_set(value);
}

float MedianAggregateSet::get() {
  return boost::accumulators::extract::median(this->median_acc_set);
}

NUniqueAggregateSet::NUniqueAggregateSet(){
  this->values_set = std::unordered_set<float>{};
}

void NUniqueAggregateSet::set_value(float value) {
  this->values_set.insert(value);
}

float NUniqueAggregateSet::get() {
  return this->values_set.size();
}

DuplicatesAggregateSet::DuplicatesAggregateSet(){
  this->values_set = std::unordered_set<float>{};
  this->duplicates = 0;
}

void DuplicatesAggregateSet::set_value(float value) {
  if(this->values_set.find(value) != this->values_set.end()){
    this->duplicates++;
  } else {
    this->values_set.insert(value);
  }
}

float DuplicatesAggregateSet::get() {
  return this->duplicates;
}

LastAggregateSet::LastAggregateSet(){
  this->last_value = std::nanf("");
}

void LastAggregateSet::set_value(float value) {
  this->last_value = value;
}

float LastAggregateSet::get() {
  return this->last_value;
}

ModeAggregateSet::ModeAggregateSet(){
  this->value_num_map = std::unordered_map<float, int>{};
  this->mode = std::nanf("");
  this->mode_freq = 0;
}

void ModeAggregateSet::set_value(float value) {
  int current_freq;
  if(this->value_num_map.find(value) == this->value_num_map.end()){
    current_freq = (this->value_num_map[value] = 1);
  } else {
    current_freq = (++this->value_num_map[value]);
  }
  if (current_freq >= this->mode_freq) {
    this->mode = value;
    this->mode_freq = current_freq;
  }
}

float ModeAggregateSet::get() {
  return this->mode;
}

ModeRatioAggregateSet::ModeRatioAggregateSet(){
  this->value_num_map = std::unordered_map<float, int>{};
  this->count = 0;
  this->mode_freq = 0;
}

void ModeRatioAggregateSet::set_value(float value) {
  int current_freq;
  this->count++;
  if(this->value_num_map.find(value) == this->value_num_map.end()){
    current_freq = (this->value_num_map[value] = 1);
  } else {
    current_freq = (++this->value_num_map[value]);
  }
  if (current_freq >= this->mode_freq) {
    this->mode_freq = current_freq;
  }
}

float ModeRatioAggregateSet::get() {
  return (float)this->mode_freq / (float)count;
}

KurtosisAggregateSet::KurtosisAggregateSet(){
  this->kurtosis_acc_set = KURTOSIS_ACC_SET{};
}

void KurtosisAggregateSet::set_value(float value) {
  this->kurtosis_acc_set(value);
}

float KurtosisAggregateSet::get() {
  return boost::accumulators::extract::kurtosis(this->kurtosis_acc_set);
}

SkewnessAggregateSet::SkewnessAggregateSet(){
  this->skewness_acc_set = SKEWNESS_ACC_SET{};
}

void SkewnessAggregateSet::set_value(float value) {
  this->skewness_acc_set(value);
}

float SkewnessAggregateSet::get() {
  return boost::accumulators::extract::skewness(this->skewness_acc_set);
}

RollingSum10AggregateSet::RollingSum10AggregateSet(){
  this->rolling_queue = std::queue<float>{};
  this->sum = 0;
}

void RollingSum10AggregateSet::set_value(float value) {
  if(this->rolling_queue.size() >= 10){
    this->sum -= this->rolling_queue.front();
    this->rolling_queue.pop();
  }
  this->sum += value;
  this->rolling_queue.push(value);
}

float RollingSum10AggregateSet::get() {
  return this->sum;
}

RollingMean10AggregateSet::RollingMean10AggregateSet(){
  this->rolling_queue = std::queue<float>{};
  this->sum = 0;
}

void RollingMean10AggregateSet::set_value(float value) {
  if(this->rolling_queue.size() >= 10){
    this->sum -= this->rolling_queue.front();
    this->rolling_queue.pop();
  }
  this->sum += value;
  this->rolling_queue.push(value);
}

float RollingMean10AggregateSet::get() {
  if(this->rolling_queue.size() > 0) {
    return this->sum / this->rolling_queue.size();
  } else {
    return std::nanf("");
  }
}

Delay10AggregateSet::Delay10AggregateSet(){
  this->rolling_queue = std::queue<float>{};
}

void Delay10AggregateSet::set_value(float value) {
  if(this->rolling_queue.size() >= 10){
    this->rolling_queue.pop();
  }
  this->rolling_queue.push(value);
}

float Delay10AggregateSet::get() {
  if(this->rolling_queue.size() >= 10){
    return this->rolling_queue.front();
  } else {
    return std::nanf("");
  }
}
