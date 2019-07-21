inv_type_memo = {}


class RelationType():
    def __init__(self, str, src_is_unique, dst_is_unique):
        self.str = str
        self.__src_is_unique = src_is_unique
        self.__dst_is_unique = dst_is_unique

    def __repr__(self):
        return self.str

    def __hash__(self):
        return hash(self.str)

    def __eq__(self, other):
        return self.str == other.str

    @property
    def inverse(self):
        return inv_type_memo[self]

    @property
    def src_is_unique(self):
        return self.__src_is_unique

    @property
    def dst_is_unique(self):
        return self.__dst_is_unique


one_to_one = RelationType("one_to_one", True, True)
one_to_many = RelationType("one_to_many", True, False)
many_to_one = RelationType("many_to_one", False, True)
many_to_many = RelationType("many_to_many", False, False)

inv_type_memo[one_to_one] = one_to_one
inv_type_memo[one_to_many] = many_to_one
inv_type_memo[many_to_one] = one_to_many
inv_type_memo[many_to_many] = many_to_many

automl_relation_type_to_relation_type = {
    'one_to_one': one_to_one,
    'one_to_many': one_to_many,
    'many_to_one': many_to_one,
    'many_to_many': many_to_many
}
