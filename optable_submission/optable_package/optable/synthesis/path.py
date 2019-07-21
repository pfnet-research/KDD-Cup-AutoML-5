import copy

import numpy as np

import optable.dataset.relation as relation_mod
import optable.dataset.relation_types as relation_types


class Path(object):
    def __init__(self, relations, root=None):
        self.__relations = relations
        if root is None:
            self.__root = relations[0].src
        else:
            self.__root = root
            if len(relations) > 0 and relations[0].src != root:
                raise Exception("relations[0].src != root")
        for r in relations:
            if not isinstance(r, relation_mod.Relation):
                raise Exception("{} is not relation.".format(r))
        for i in range(len(relations) - 1):
            if relations[i].dst != relations[i+1].src:
                raise Exception("{}.dst is not {}.src".format(
                    relations[i], relations[i+1]))
        self.__depth_of_path = None
        self.__deeper_count = None
        self.__not_deeper_count = None
        self.__shallower_count = None
        self.__not_shallower_count = None

        self.__to_one_count = 0
        self.__to_many_count = 0
        self.__self_mm_count = 0
        self.__mm_count = 0
        for r in self.relations:
            if r.type == relation_types.many_to_many:
                self.__mm_count += 1
            if r.type == relation_types.one_to_one \
               or r.type == relation_types.many_to_one:
                self.__to_one_count += 1
            if r.src == r.dst and r.src_id == r.dst_id \
               and r.type == relation_types.many_to_many:
                self.__self_mm_count += 1
            if r.type == relation_types.one_to_many \
               or r.type == relation_types.many_to_many:
                self.__to_many_count += 1

        if len(self) > 0 and self.relations[0].src == self.relations[0].dst \
           and self.relations[0].src_id == self.relations[0].dst_id \
           and self.relations[0].type == relation_types.many_to_many:
            self.__first_self_mm = True
        else:
            self.__first_self_mm = False

        self.__nodes = [self.src] + [r.dst for r in self.relations]
        self.__is_unique_path = (len(self.__nodes) == len(set(self.__nodes)))

    def __copy__(self):
        return Path(copy.copy(self.relations), self.root)

    def __repr__(self):
        ret = "Path(\n"
        ret += "from {}\n".format(self.root)
        for r in self.relations:
            ret += "{}\n".format(r)
        ret += ")\n"
        return ret

    def __hash__(self):
        return hash(tuple(self.relations))

    def __eq__(self, other):
        if other is None:
            return False
        return tuple(self.relations) == tuple(other.relations)

    def __add__(self, other):
        if isinstance(other, relation_mod.Relation):
            if self.dst != other.src:
                raise Exception("{}.dst is not {}.src".format(
                    self.relations[-1], other))
            relations = copy.copy(self.relations)
            relations.append(other)
            ret = Path(relations)
            return ret
        else:
            raise Exception("type {} is invalid.".format(type(other)))

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, index):
        if isinstance(index, slice):
            new_relations = self.relations[index]
            if index.start is not None and index.start == 0:
                new_root = self.__root
            elif (
                index.start is not None and index.start == len(self.relations)
            ):
                new_root = self.relations[index.start - 1].dst
            elif index.start is not None and index.start > 0:
                new_root = self.relations[index.start].src
            else:
                new_root = self.root
            return Path(new_relations, root=new_root)

            if index.start is not None and index.start > 0:
                return Path(self.relations[index],
                            root=self.relations[index.start - 1].dst)
            else:
                if index.stop is None:
                    return self
                else:
                    return Path(self.relations[index])
        elif isinstance(index, int):
            return self.relations[index]
        else:
            raise ValueError("Invalid index")

    def set_depths(self, depths):
        self.__depth_of_path = \
            [depths[self.src]] \
            + [depths[r.dst] for r in self.relations]

        self.__deeper_count = 0
        self.__not_deeper_count = 0
        self.__shallower_count = 0
        self.__not_shallower_count = 0
        for i in range(len(self.relations)):
            if self.depth_of_path[i + 1] > self.depth_of_path[i]:
                self.__deeper_count += 1
                self.__not_shallower_count += 1
            elif self.depth_of_path[i + 1] == self.depth_of_path[i]:
                self.__not_deeper_count += 1
                self.__not_shallower_count += 1
            else:
                self.__not_deeper_count += 1
                self.__shallower_count += 1

    @property
    def relations(self):
        return self.__relations

    @property
    def root(self):
        return self.__root

    @property
    def table_names(self):
        return [self.__root] + [rel.dst for rel in self.__relations]

    @property
    def src(self):
        if len(self) == 0:
            return self.root
        return self.relations[0].src

    @property
    def dst(self):
        if len(self) == 0:
            return self.root
        return self.relations[-1].dst

    @property
    def src_id(self):
        if len(self) == 0:
            return None
        return self.relations[0].src_id

    @property
    def dst_id(self):
        if len(self) == 0:
            return None
        return self.relations[-1].dst_id

    @property
    def nodes(self):
        return self.__nodes

    @property
    def is_to_one(self):
        ret = True
        for r in self.relations:
            if r.type == relation_types.one_to_many \
               or r.type == relation_types.many_to_many:
                ret = False
        return ret

    @property
    def is_to_many(self):
        for r in self.relations:
            if r.type == relation_types.one_to_many \
               or r.type == relation_types.many_to_many:
                return True
        return False

    @property
    def depth_of_path(self):
        return self.__depth_of_path

    @property
    def self_mm_count(self):
        return self.__self_mm_count

    @property
    def mm_count(self):
        return self.__mm_count

    @property
    def to_one_count(self):
        return self.__to_one_count

    @property
    def to_many_count(self):
        return self.__to_many_count

    @property
    def deeper_count(self):
        return self.__deeper_count

    @property
    def not_deeper_count(self):
        return self.__not_deeper_count

    @property
    def shallower_count(self):
        return self.__shallower_count

    @property
    def not_shallower_count(self):
        return self.__not_shallower_count

    @property
    def first_self_mm(self):
        return self.__first_self_mm

    @property
    def is_unique_path(self):
        return self.__is_unique_path

    def is_last_to_many(self, idx):
        r = self.relations[idx]
        if r.type != relation_types.one_to_one \
           and r.type != relation_types.many_to_one:
            return False
        for i in range(idx + 1, len(self)):
            r = self.relations[i]
            if r.type != relation_types.one_to_many \
               and r.type != relation_types.many_to_many:
                return False
        return True

    def is_to_one_after_last_to_many(self, idx):
        r = self.relations[idx]
        if r.type != relation_types.one_to_many \
           and r.type != relation_types.many_to_many:
            return False
        for i in range(idx + 1, len(self)):
            r = self.relations[i]
            if r.type != relation_types.one_to_many \
               and r.type != relation_types.many_to_many:
                return False
        return True

    def is_substance_to_one_with_col(self, dataset, col):
        if len(self.relations) >= 1:
            for r1, r2 in zip(self.relations[:-1], self.relations[1:]):
                if r1.type == relation_types.one_to_many \
                   or r1.type == relation_types.many_to_many:
                    if dataset.tables[r1.dst].has_cache(
                        ('substance_to_one_columns', r1.dst_id)
                    ):
                        if not dataset.tables[r1.dst].get_cache(
                            ('substance_to_one_columns', r1.dst_id)
                        )[r2.src_id]:
                            return False
                    else:
                        return False
            last_r = self.relations[-1]
            if last_r.type == relation_types.one_to_many \
               or last_r.type == relation_types.many_to_many:
                if dataset.tables[last_r.dst].has_cache(
                    ('substance_to_one_columns', last_r.dst_id)
                ):
                    if not dataset.tables[last_r.dst].get_cache(
                        ('substance_to_one_columns', last_r.dst_id)
                    )[col]:
                        return False
                else:
                    return False
        return True

    def to_many_nunique(self, dataset):
        nunique = 0
        for relation in self.relations:
            nunique += dataset.tables[relation.dst].nunique[relation.dst_id]
        return nunique

    def to_many_path_priority(self, dataset, col):
        path_to_many_nunique = self.to_many_nunique(dataset)
        data_nunique = dataset.tables[self.dst].nunique[col]
        path_table_len = np.sum([
            len(dataset.tables[relation.dst].df)
            for relation in self.relations])
        dst_table_len = len(dataset.tables[self.dst].df)
        nunique_priority = 1 - (
            (np.log1p(path_to_many_nunique) + np.log1p(data_nunique))
            / (np.log1p(path_table_len) + np.log1p(path_table_len))
        )
        return nunique_priority

    def substance_to_many_count(self, dataset, col=None):
        ret = 0
        if len(self.relations) >= 1:
            for r1, r2 in zip(self.relations[:-1], self.relations[1:]):
                if r1.type == relation_types.one_to_many \
                   or r1.type == relation_types.many_to_many:
                    if dataset.tables[r1.dst].has_cache(
                        ('substance_to_one_columns', r1.dst_id)
                    ):
                        if not dataset.tables[r1.dst].get_cache(
                            ('substance_to_one_columns', r1.dst_id)
                        )[r2.src_id]:
                            ret += 1
                    else:
                        ret += 1
            last_r = self.relations[-1]
            if last_r.type == relation_types.one_to_many \
               or last_r.type == relation_types.many_to_many:
                if col is not None:
                    if dataset.tables[last_r.dst].has_cache(
                        ('substance_to_one_columns', last_r.dst_id)
                    ):
                        if not dataset.tables[last_r.dst].get_cache(
                            ('substance_to_one_columns', last_r.dst_id)
                        )[col]:
                            ret += 1
                    else:
                        ret += 1
                else:
                    ret += 1
        return ret
