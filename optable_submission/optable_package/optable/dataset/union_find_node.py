# https://qiita.com/maron8676/items/6010128f4ef5c268f719


class UnionFindNode(object):
    def __init__(self, group_id, parent=None, value=None):
        self.group_id_ = group_id
        self.parent_ = parent
        self.value = value
        self.rank_ = 1

    def __str__(self):
        template = \
            "UnionFindNode(group_id: {}, \n\tparent: {}, value: {}, size: {})"
        return template.format(
            self.group_id_, self.parent_, self.value, self.rank_)

    @property
    def group_id(self):
        return self.group_id_

    def is_root(self):
        return not self.parent_

    def root(self):
        parent = self
        while not parent.is_root():
            parent = parent.parent_
            self.parent_ = parent
        return parent

    def find(self):
        root = self.root()
        return root.group_id_

    def rank(self):
        root = self.root()
        return root.rank_

    def unite(self, unite_node):
        root = self.root()
        unite_root = unite_node.root()

        if root.group_id_ != unite_root.group_id_:
            if root.rank() > unite_root.rank():
                unite_root.parent_ = root
                root.rank_ = max(root.rank_, unite_root.rank_ + 1)
            else:
                root.parent_ = unite_root
                unite_root.rank_ = max(root.rank_ + 1, unite_root.rank_)
