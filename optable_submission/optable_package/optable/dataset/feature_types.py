class FeatureType():
    def __init__(self, type_name, prefix):
        self.__type_name = type_name
        self.__prefix = prefix

    def __repr__(self):
        # -> str
        return self.__type_name

    def __hash__(self):
        # -> int
        return hash(self.__type_name)

    def __eq__(self, other: object):
        # -> boolean
        assert isinstance(other, FeatureType)
        return self.__type_name == other.type_name

    @property
    def type_name(self):
        return self.__type_name

    @property
    def prefix(self):
        return self.__prefix


COMMON = FeatureType("COMMON", None)
unknown = FeatureType("unknown", None)  # TODO(yoshikawa): 廃止
numerical = FeatureType("numerical", "n_")
categorical = FeatureType("categorical", "c_")
time = FeatureType("time", "t_")
multi_categorical = FeatureType("multi_categorical", "m_")
mc_processed_numerical = FeatureType("mc_processed_numerical", "mpn_")
c_processed_numerical = FeatureType("c_processed_numerical", "cpn_")
t_processed_numerical = FeatureType("t_processed_numerical", "tpn_")
n_processed_categorical = FeatureType("n_processed_categorical", "npc_")
mc_processed_categorical = FeatureType("mc_processed_categorical", "mpc_")
c_processed_categorical = FeatureType("c_processed_categorical", "cpc_")
t_processed_categorical = FeatureType("t_processed_categorical", "tpc_")
aggregate_processed_numerical = \
    FeatureType("aggregate_processed_numerical", "apn_")
aggregate_processed_categorical = \
    FeatureType("aggregate_processed_categorical", "apc_")

ftypes = [numerical, categorical, time, multi_categorical,
          mc_processed_numerical, c_processed_numerical, t_processed_numerical,
          n_processed_categorical, mc_processed_categorical,
          c_processed_categorical, t_processed_categorical,
          aggregate_processed_numerical, aggregate_processed_categorical]

prefix_to_ftype = {ftype.prefix: ftype for ftype in ftypes}


def column_name_to_ftype(column_name):
    prefix = "{}_".format(column_name.split("_")[0])
    return prefix_to_ftype[prefix]
