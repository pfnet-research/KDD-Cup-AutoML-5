from optable.dataset import relation_types


class Relation(object):
    def __init__(self, src, dst, src_id, dst_id, type):
        self.__src = src
        self.__dst = dst
        self.__src_id = src_id
        self.__dst_id = dst_id
        if isinstance(type, str):
            self.__type = \
                relation_types.automl_relation_type_to_relation_type[type]
        elif isinstance(type, relation_types.RelationType):
            self.__type = type
        else:
            raise ValueError("invalid relation type {}".format(type))

    def __repr__(self):
        return "%s[%s]-[%s]->[%s]%s" \
            % (self.src, self.src_id, self.type, self.dst_id, self.dst)

    def __hash__(self):
        return hash((self.src, self.dst,
                     self.src_id, self.dst_id, self.type))

    def __eq__(self, other):
        if isinstance(other, Relation):
            return (self.src == other.src) \
                and (self.dst == other.dst) \
                and (self.src_id == other.src_id) \
                and (self.dst_id == other.dst_id) \
                and (self.type == other.type)
        else:
            return False

    @property
    def inverse(self):
        return Relation(src=self.dst, dst=self.src, src_id=self.dst_id,
                        dst_id=self.src_id, type=self.type.inverse)

    @property
    def src(self):
        return self.__src

    @property
    def dst(self):
        return self.__dst

    @property
    def src_id(self):
        return self.__src_id

    @property
    def dst_id(self):
        return self.__dst_id

    @property
    def type(self):
        return self.__type
