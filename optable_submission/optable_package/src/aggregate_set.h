#pragma once

#include <pybind11/stl.h> // vector用
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include <queue>

typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::sum>
  > SUM_ACC_SET;
typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::variance>
  > VARIANCE_ACC_SET;
typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::max>
  > MAX_ACC_SET;
typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::min>
  > MIN_ACC_SET;
typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::mean>
  > MEAN_ACC_SET;
typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::median>
  > MEDIAN_ACC_SET;
typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::kurtosis>
  > KURTOSIS_ACC_SET;
typedef boost::accumulators::accumulator_set<
  float, boost::accumulators::stats<boost::accumulators::tag::skewness>
  > SKEWNESS_ACC_SET;

class AggregateSet{
  public:
    virtual void set_value(float value){};
    virtual float get() {return std::nanf("");};
};

class SumAggregateSet : public AggregateSet{
private:
  SUM_ACC_SET sum_acc_set;
public:
  SumAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class VarianceAggregateSet : public AggregateSet{
private:
  VARIANCE_ACC_SET variance_acc_set;
public:
  VarianceAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class MaxAggregateSet : public AggregateSet{
private:
  MAX_ACC_SET max_acc_set;
public:
  MaxAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class MinAggregateSet : public AggregateSet{
private:
  MIN_ACC_SET min_acc_set;
public:
  MinAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class MeanAggregateSet : public AggregateSet{
private:
  MEAN_ACC_SET mean_acc_set;
public:
  MeanAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class MedianAggregateSet : public AggregateSet{
private:
  MEDIAN_ACC_SET median_acc_set;
public:
  MedianAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class NUniqueAggregateSet : public AggregateSet{
private:
  std::unordered_set<float> values_set;
public:
  NUniqueAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class DuplicatesAggregateSet : public AggregateSet{
private:
  std::unordered_set<float> values_set;
  int duplicates;
public:
  DuplicatesAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class LastAggregateSet : public AggregateSet{
private:
  float last_value;
public:
  LastAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class ModeAggregateSet : public AggregateSet{
private:
  std::unordered_map<float, int> value_num_map;
  float mode;
  int mode_freq;
public:
  ModeAggregateSet();
  void set_value(float value) override;
  float get() override;  
};

class ModeRatioAggregateSet : public AggregateSet{
private:
  std::unordered_map<float, int> value_num_map;
  int count;
  int mode_freq;
public:
  ModeRatioAggregateSet();
  void set_value(float value) override;
  float get() override;  
};

class KurtosisAggregateSet : public AggregateSet{
private:
  KURTOSIS_ACC_SET kurtosis_acc_set;
public:
  KurtosisAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class SkewnessAggregateSet : public AggregateSet{
private:
  SKEWNESS_ACC_SET skewness_acc_set;
public:
  SkewnessAggregateSet();
  void set_value(float value) override;
  float get() override;
};

class RollingSum10AggregateSet : public AggregateSet{
private:
  std::queue<float> rolling_queue;
  float sum;
public:
  RollingSum10AggregateSet();
  void set_value(float value) override;
  float get() override;
};

class RollingMean10AggregateSet : public AggregateSet{
private:
  std::queue<float> rolling_queue;
  float sum;
public:
  RollingMean10AggregateSet();
  void set_value(float value) override;
  float get() override;
};

class Delay10AggregateSet : public AggregateSet{
private:
  std::queue<float> rolling_queue;
public:
  Delay10AggregateSet();
  void set_value(float value) override;
  float get() override;
};
