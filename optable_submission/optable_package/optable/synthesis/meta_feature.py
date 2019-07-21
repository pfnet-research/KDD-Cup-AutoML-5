import numpy as np

from optable import manipulations  # NOQA
from optable.synthesis import manipulation_candidate


meta_feature_start = {}
meta_feature_names = []
current_start = 0
for manipulation_class in manipulation_candidate.manipulation_classes:
    meta_feature_start[manipulation_class] = current_start
    current_start += manipulation_class.meta_feature_size()
    meta_feature_names += manipulation_class.meta_feature_name()
total_meta_feature_size = current_start


def calculate_meta_feature(manipulation):
    meta_feature = np.zeros(total_meta_feature_size)

    if not type(manipulation) in meta_feature_start:
        print("invalid manipulation type", type(manipulation))
        return meta_feature
    start_ = meta_feature_start[type(manipulation)]
    size_ = type(manipulation).meta_feature_size()
    meta_feature_data = manipulation.meta_feature()
    assert(size_ == len(meta_feature_data))
    meta_feature[start_: start_+size_] = meta_feature_data
    return meta_feature
