# username - yotamzvieli
# id1      - 209497023
# name1    - Yotam Zvieli
# id2      - 208410084
# name2    - Nofar Shlomo


"""A class represnting a node in an AVL tree"""


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type value: str
    @param value: data of your node
    """

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1
        self.size = 0
        self.balance_factor = 0

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        return self.value if self.height != -1 else None

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height

    """sets left child

    @type node: AVLNode
    @param node: a node
    """

    def setLeft(self, node):
        self.left = node
        self.update_node_fields()

    """sets right child

    @type node: AVLNode
    @param node: a node
    """

    def setRight(self, node):
        self.right = node
        self.update_node_fields()

    """sets parent

    @type node: AVLNode
    @param node: a node
    """

    def setParent(self, node):
        self.parent = node

    """sets value

    @type value: str
    @param value: data
    """

    def setValue(self, value):
        self.value = value

    """sets the balance factor of the node

    @type h: int
    @param h: the height
    """

    def setHeight(self, h):
        self.height = h

    """returns whether self is not a virtual node 

    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def isRealNode(self):
        return self.height != -1

    """
    update size, height and balance factor node fields
    """

    def update_node_fields(self):
        self.size = self.left.size + self.right.size + 1
        self.height = max(self.left.height, self.right.height) + 1
        self.balance_factor = self.left.height - self.right.height


"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self,root=None):
        self.size = 0
        self.root = root

    # add your fields here

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        return self.size == 0

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    """

    def retrieve(self, i):
        if(self.root.left.size > i):
            l = AVLTreeList(self.root.left)
            return l.retrieve(i)
        elif(self.root.left.size == i):
            return self.root.value
        else:
            l = AVLTreeList(self.root.right)
            left_size = self.root.left.size
            return l.retrieve(i-left_size)


    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def insert(self, i, val):
        rotate_count = 0
        node_to_insert = self.generate_new_node(val)
        self.insert_node(node_to_insert, i)
        curr_node = node_to_insert
        while curr_node != None:
            curr_node.update_node_fields()
            if abs(curr_node.balance_factor) > 1:
                self.rebalance_and_update(curr_node)
                rotate_count += 1
            curr_node = curr_node.parent
        return rotate_count

    def rebalance_and_update(self, node):
        if node.balance_factor == 2 and node.left.balance_factor == 1:
            node_positive_two = node
            node_positive_one = node.left
            parent = node.parent
            if parent.left == node_positive_two:
                parent.left = node_positive_one
            else:
                parent.right = node_positive_one
            node_positive_one.parent = node_positive_two.parent
            node_positive_one.right = node_positive_two
            node_positive_two.parent = node_positive_one
            node_positive_two.left = node_positive_one.right
            node_positive_two.left.parent = node_positive_two
        elif node.balance_factor == -2 and node.right.balance_factor == -1:
            node_neg_two = node
            node_neg_one = node.right
            parent = node.parent
            if parent.right == node_neg_two:
                parent.right = node_neg_one
            else:
                parent.left = node_neg_one



    def generate_new_node(self, val):
        node_to_insert = AVLNode(val)
        right_vir = AVLNode(None)
        left_vir = AVLNode(None)
        node_to_insert.left = left_vir
        left_vir.parent = node_to_insert
        right_vir.parent = node_to_insert
        node_to_insert.right = right_vir
        node_to_insert.update_node_fields()
        return node_to_insert



    def insert_node(self,node,i):
        if i == 0:
            node_to_insert_before = self.retrieve(0)
            node_to_insert_before.left = node
            node.parent = node_to_insert_before
        else:
            node_to_insert_after = self.retrieve(i-1)
            if node_to_insert_after.right.isRealNode:
                successor = self.succesor(node_to_insert_after)
                successor.left = node
                node.parent = successor
            else:
                node_to_insert_after.right = node
                node.parent = node_to_insert_after

    def succesor(self,node):
        return None

    """deletes the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, i):
        return -1

    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        return None if self.empty() else self.retrieve(0)

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        return None if self.empty() else self.retrieve(self.length()-1)

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    """

    def listToArray(self):
        return None

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        return self.root.size

    """sort the info values of the list

    @rtype: list
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    """

    def sort(self):
        return None

    """permute the info values of the list 

    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    """

    def permutation(self):
        return None

    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst):
        return None

    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val):
        return None

    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return None
