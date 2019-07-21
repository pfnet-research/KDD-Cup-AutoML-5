from optable.synthesis import manipulation_candidate

from optable.manipulations.aggregations import count
from optable.manipulations.aggregations import duplicates
from optable.manipulations.aggregations import identify
from optable.manipulations.aggregations import last
from optable.manipulations.aggregations import max
from optable.manipulations.aggregations import mean_variance
from optable.manipulations.aggregations import mean
from optable.manipulations.aggregations import median
from optable.manipulations.aggregations import min
from optable.manipulations.aggregations import mode
from optable.manipulations.aggregations import nunique
from optable.manipulations.aggregations import one_hot_mean
from optable.manipulations.aggregations import one_hot_sum
from optable.manipulations.aggregations import sum_variance
from optable.manipulations.aggregations import sum

from optable.manipulations.factorized import main_factorized_numerical

from optable.manipulations.target_encodings import factorized_target_encoding
from optable.manipulations.target_encodings \
    import hist_neighborhood_target_encoding
from optable.manipulations.target_encodings \
    import multi_categorical_target_encoding
from optable.manipulations.target_encodings import target_encoding

from optable.manipulations.time_features import max_diff_min_time
from optable.manipulations.time_features import sequential_regression
from optable.manipulations.time_features import time_diff_in_same_table
from optable.manipulations.time_features import time_diff


manipulation_candidate.register_manipulation_candidate(
    count.CountCandidate(), count.CountManipulation)
manipulation_candidate.register_manipulation_candidate(
    duplicates.DuplicatesCandidate(), duplicates.DuplicatesManipulation)
manipulation_candidate.register_manipulation_candidate(
    identify.IdentifyCandidate(), identify.IdentifyManipulation)
manipulation_candidate.register_manipulation_candidate(
    last.LastCandidate(), last.LastManipulation)
manipulation_candidate.register_manipulation_candidate(
    max.MaxCandidate(), max.MaxManipulation)
manipulation_candidate.register_manipulation_candidate(
    mean_variance.MeanVarianceCandidate(),
    mean_variance.MeanVarianceManipulation)
manipulation_candidate.register_manipulation_candidate(
    mean.MeanCandidate(), mean.MeanManipulation)
manipulation_candidate.register_manipulation_candidate(
    median.MedianCandidate(), median.MedianManipulation)
manipulation_candidate.register_manipulation_candidate(
    min.MinCandidate(), min.MinManipulation)
manipulation_candidate.register_manipulation_candidate(
    mode.ModeCandidate(), mode.ModeManipulation)
manipulation_candidate.register_manipulation_candidate(
    nunique.NuniqueCandidate(), nunique.NuniqueManipulation)
manipulation_candidate.register_manipulation_candidate(
    one_hot_mean.OneHotMeanCandidate(), one_hot_mean.OneHotMeanManipulation)
manipulation_candidate.register_manipulation_candidate(
    one_hot_sum.OneHotSumCandidate(), one_hot_sum.OneHotSumManipulation)
manipulation_candidate.register_manipulation_candidate(
    sum_variance.SumVarianceCandidate(), sum_variance.SumVarianceManipulation)
manipulation_candidate.register_manipulation_candidate(
    sum.SumCandidate(), sum.SumManipulation)

manipulation_candidate.register_manipulation_candidate(
    main_factorized_numerical.MainFactorizedNumericalCandidate(),
    main_factorized_numerical.MainFactorizedNumericalManipulation)

manipulation_candidate.register_manipulation_candidate(
    factorized_target_encoding.FactorizedTargetEncodingCandidate(),
    factorized_target_encoding.FactorizedTargetEncodingManipulation)
manipulation_candidate.register_manipulation_candidate(
    hist_neighborhood_target_encoding
    .HistNeighborhoodTargetEncodingCandidate(),
    hist_neighborhood_target_encoding
    .HistNeighborhoodTargetEncodingManipulation
)
manipulation_candidate.register_manipulation_candidate(
    multi_categorical_target_encoding
    .MultiCategoricalTargetEncodingCandidate(),
    multi_categorical_target_encoding
    .MultiCategoricalTargetEncodingManipulation
)
manipulation_candidate.register_manipulation_candidate(
    target_encoding.TargetEncodingCandidate(),
    target_encoding.TargetEncodingManipulation)

manipulation_candidate.register_manipulation_candidate(
    max_diff_min_time.MaxDiffMinCandidate(),
    max_diff_min_time.MaxDiffMinManipulation)
manipulation_candidate.register_manipulation_candidate(
    sequential_regression.SequentialRegressionCandidate(),
    sequential_regression.SequentialRegressionManipulation)
manipulation_candidate.register_manipulation_candidate(
    time_diff_in_same_table.TimeDiffInSameTableCandidate(),
    time_diff_in_same_table.TimeDiffInSameTableManipulation)
manipulation_candidate.register_manipulation_candidate(
    time_diff.TimeDiffCandidate(),
    time_diff.TimeDiffManipulation)
