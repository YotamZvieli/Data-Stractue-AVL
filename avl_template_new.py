# username - yotamzvieli
# id1      - 209497023
# name1    - Yotam Zvieli
# id2      - 208410084
# name2    - Nofar Shlomo


"""A class represnting a node in an AVL tree"""
import random

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

    def getSize(self):
        return self.size

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

    def update_node_fields(self): #Func to update node fileds - O(1)
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

    def __init__(self, root=None):
        self.size = 0
        self.root = root
        self.firstItem = None
        self.lastItem = None

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

    def retrieve_node(self, i): #rec func to retrieve node O(log n) run time
        if (self.root.left.size > i):
            l = AVLTreeList(self.root.left)
            return l.retrieve_node(i)
        elif (self.root.left.size == i):
            return self.root
        else:
            l = AVLTreeList(self.root.right)
            left_size = self.root.left.size
            return l.retrieve_node(i - left_size - 1)

    """retrieves the value of the i'th item in the list

        @type i: int
        @pre: 0 <= i < self.length()
        @param i: index in the list
        @rtype: AVLNode
        @returns: the node with the i's value
        """
    def retrieve(self, i): #invelope func for retrieve node O(1) run time
        if (i >= self.length() or i < 0):
            return None
        return self.retrieve_node(i).value

    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def insert(self, i, val): #insert node O(log n)
        if (self.root == None):
            self.root = self.generate_new_node(val)
            self.root.update_node_fields()
            self.size = self.root.size
            return 0
        rotate_count = 0
        node_to_insert = self.generate_new_node(val)
        self.insert_node(node_to_insert, i)
        curr_node = node_to_insert
        while curr_node != None:
            curr_node.update_node_fields()
            if abs(curr_node.balance_factor) > 1:
                curr_node, temp_cnt = self.rebalance_and_update(curr_node, rotate_count)
                rotate_count += temp_cnt
            else:
                curr_node.update_node_fields()
            prev = curr_node
            curr_node = curr_node.parent
        self.size = prev.size
        self.root = prev
        self.update_firls_last()
        return rotate_count

    """rebalanced and update a specific node position if BF > 1 -  O(1) rum time

        @type node: AVLNode
        @type cnt: int
        @pre: abs(node.balance factor) > 1 
        @rtype: int
        @returns: cnt  + num of rotate operations 
        """
    def rebalance_and_update(self, node, cnt):
        isRoot = True if node.parent == None else False
        if node.balance_factor == 2 and (node.left.balance_factor == 1 or node.left.balance_factor == 0):
            node_positive_two = node
            node_positive_one = node.left
            parent = node.parent
            if not (isRoot):
                if parent.left == node_positive_two:
                    parent.left = node_positive_one
                else:
                    parent.right = node_positive_one
            node_positive_one.parent = node_positive_two.parent
            node_positive_two.left = node_positive_one.right
            node_positive_one.right = node_positive_two
            node_positive_two.parent = node_positive_one
            node_positive_two.left.parent = node_positive_two
            node_positive_two.update_node_fields()
            node_positive_one.update_node_fields()
            cnt += 1
            return (node_positive_one, cnt)
        elif node.balance_factor == -2 and (node.right.balance_factor == -1 or node.right.balance_factor == 0):
            node_neg_two = node
            node_neg_one = node.right
            parent = node.parent
            if not (isRoot):
                if parent.right == node_neg_two:
                    parent.right = node_neg_one
                else:
                    parent.left = node_neg_one
            node_neg_two.right = node_neg_one.left
            node_neg_one.left = node_neg_two
            node_neg_one.parent = node_neg_two.parent
            node_neg_two.parent = node_neg_one
            node_neg_two.right.parent = node_neg_two
            node_neg_two.update_node_fields()
            node_neg_one.update_node_fields()
            cnt += 1
            return (node_neg_one, cnt)
        elif node.balance_factor == -2 and node.right.balance_factor == 1:
            node_neg_two = node
            node_pos_one = node.right
            node_zero = node.right.left
            parent = node.parent
            if not (isRoot):
                if parent.right == node_neg_two:
                    parent.right = node_zero
                else:
                    parent.left = node_zero
            node_neg_two.parent = node_zero
            node_neg_two.right = node_zero.left
            node_pos_one.parent = node_zero
            node_pos_one.left = node_zero.right
            node_zero.right.parent = node_pos_one
            node_zero.left.parent = node_neg_two
            node_zero.left = node_neg_two
            node_zero.right = node_pos_one
            node_zero.parent = parent
            node_neg_two.update_node_fields()
            node_pos_one.update_node_fields()
            node_zero.update_node_fields()
            cnt += 2
            return (node_zero, cnt)
        elif node.balance_factor == 2 and node.left.balance_factor == -1:
            node_pos_two = node
            node_neg_one = node.left
            node_zero = node.left.right
            parent = node.parent
            if not (isRoot):
                if parent.right == node_pos_two:
                    parent.right = node_zero
                else:
                    parent.left = node_zero
            node_pos_two.parent = node_zero
            node_pos_two.left = node_zero.right
            node_zero.right.parent = node_pos_two
            node_neg_one.parent = node_zero
            node_neg_one.right = node_zero.left
            node_zero.left.parent = node_neg_one
            node_zero.left = node_neg_one
            node_zero.right = node_pos_two
            node_zero.parent = parent
            node_pos_two.update_node_fields()
            node_neg_one.update_node_fields()
            node_zero.update_node_fields()
            cnt += 2
            return (node_zero, cnt)

    """generate new node to insert with virtual suns - o(1) run time

            @type val: str
            @rtype: AVLNode
            """
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

    """insert the node and return the node to start rebalanced and update - O(1)

       @type i: int
       @pre: 0 <= i <= self.length()
       @rtype: None
       """
    def insert_node(self, node, i):
        if i == 0:
            node_to_insert_before = self.retrieve_node(0)
            node_to_insert_before.left = node
            node.parent = node_to_insert_before
        else:
            node_to_insert_after = self.retrieve_node(i - 1)
            if node_to_insert_after.right.isRealNode():
                successor = self.succesor(node_to_insert_after)
                successor.left = node
                node.parent = successor
            else:
                node_to_insert_after.right = node
                node.parent = node_to_insert_after

    """find successor -  O(log n) run time 

           @type node: AVLNode
           @rtype: AVLNode
           """
    def succesor(self, node):
        curr = node
        if (node.right.isRealNode()):
            return self.min(node.right)
        else:
            while curr.parent == curr.parent.right and curr.parent != None:
                curr = curr.parent
            return curr.parent

    """find first item in sub list -  O(log n) run time 

               @type node: AVLNode
               @pre node is real node
               @rtype: AVLNode
               """
    def min(self, node):
        while node.left.isRealNode():
            node = node.left
        return node

    """deletes the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    """

    def delete(self, i): # o(log n) - one way trip update files and rotateְ
        if (i < 0 or i >= self.length()):
            return -1
        if (self.length() == 1):
            self.root = None
            self.size = 0
            self.firstItem = None
            self.lastItem = None
            return 0
        parent_deleted = self.delete_node(i)
        if (parent_deleted == None):
            self.size = self.root.size
            self.update_firls_last()
            return 0
        rotate_count = 0
        curr_node = parent_deleted
        while curr_node != None:
            curr_node.update_node_fields()
            if abs(curr_node.balance_factor) > 1:
                curr_node, temp_cnt = self.rebalance_and_update(curr_node, rotate_count)
                rotate_count += temp_cnt
            else:
                curr_node.update_node_fields()
            prev = curr_node
            curr_node = curr_node.parent
        self.size = prev.size
        self.root = prev
        self.update_firls_last()
        return rotate_count

    """O(log n) - find and delete the node (replace with successor if needed)

        @type i: int
        @pre: 0 <= i < self.length()
        @param i: The intended index in the list to be deleted
        @rtype: AVLNode
        @returns: parent of Node that actually deleted from the tree - to start rebalance from.
        """
    def delete_node(self, i):
        node_to_del = self.retrieve_node(i)
        parent = self.del_simple_case(node_to_del)
        if (parent == None or parent.isRealNode()):
            return parent
        else:
            successor = self.succesor(node_to_del)
            node_to_del.value = successor.value
            return self.del_simple_case(successor)

    """
            @type node_to_del: AVLNode
            @pre: 0 <= i < self.length()
            @param i: The intended index in the list to be deleted
            @rtype: AVLNode if delete else None (func description in doc file)
            """
    def del_simple_case(self, node_to_del): # O(1) - pointers change
        parent = AVLNode(None)
        if (node_to_del.parent == None and (not node_to_del.left.isRealNode() or not node_to_del.right.isRealNode())):
            if (self.isLeaf(node_to_del)):
                self.root = None
                return None
            elif (node_to_del.left.isRealNode()):
                self.root = node_to_del.left
                self.root.parent = None
                return None
            else:
                self.root = node_to_del.right
                return None

        if (self.isLeaf(node_to_del)):
            parent = node_to_del.parent
            if (parent.left == node_to_del):
                parent.left = node_to_del.left  # replace node to del with virtual node
            else:
                parent.right = node_to_del.left  # replace node to del with virtual node
            node_to_del.left.parent = parent
        elif (self.isLeaf(node_to_del.left) and (
                not node_to_del.right.isRealNode())):  # left is leaf and right is virtual
            parent = node_to_del.parent
            son = node_to_del.left
            if (parent.left == node_to_del):
                parent.left = son
            else:
                parent.right = son
            son.parent = parent
        elif (self.isLeaf(node_to_del.right) and (
                not node_to_del.left.isRealNode())):  # right is leaf and left is virtual
            parent = node_to_del.parent
            son = node_to_del.right
            if (parent.left == node_to_del):
                parent.left = son
            else:
                parent.right = son
            son.parent = parent
        return parent

    """check if node is leaf
       @type node: AVLNode
       @rtype: bool
       @returns: true if leaf else false
       """
    def isLeaf(self, node):
        if (not node.isRealNode()):
            return False
        return not (node.left.isRealNode() or node.right.isRealNode())

    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        return None if self.empty() else self.retrieve_node(0).value

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        return None if self.empty() else self.retrieve_node(self.length() - 1).value

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    """

    def listToArray(self): # O(n) - in order trip
        if(self.size == 0):
            return []
        global in_order_lst
        in_order_lst = []
        self.listToArrayRec(self.root)
        return in_order_lst

    """make in order walk in the tree and load values to global list in order.
        @type node: AVLNode
        @rtype: None
        """
    def listToArrayRec(self, node):
        global in_order_lst
        if (not node.isRealNode()):
            return None
        if (self.isLeaf(node)):
            in_order_lst.append(node.value)
            return None
        else:
            self.listToArrayRec(node.left)
            in_order_lst.append(node.value)
            self.listToArrayRec(node.right)
            return None

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        if (self.root == None):
            return 0
        else:
            return self.root.size

    """sort the info values of the list

    @rtype: list
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    """

    def sort(self): #O(n*log n)
        lst = self.listToArray()
        lst = self.quick_sort(lst)
        return self.set_new_val_rec(lst, 0, len(lst)-1)

    def quick_sort(self, lst): #O(nlog n)
        if (len(lst) <= 1):
            return lst
        else:
            rand = random
            pivot = rand.randint(0, len(lst) - 1)
            temp = lst[pivot]
            lst[pivot] = lst[-1]
            lst[-1] = temp
            bigger = [i for i in lst if i > temp]
            equal = [i for i in lst if i == temp]
            smaller = [i for i in lst if i < temp]
            return self.quick_sort(smaller) + equal + self.quick_sort(bigger)

    """permute the info values of the list 

    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    """

    def permutation(self): #O(n)
        lst = self.listToArray()
        random_lst = []
        for i in range(self.length()):
            index = self.generate_index(len(lst) - 1 - i)
            random_lst.append(lst[index])
            lst[index] = lst[len(lst) - 1 - i]

        return self.set_new_val_rec(random_lst, 0, len(random_lst)-1)

    def set_new_val_rec(self, lst, start, end): # o(n) - Build an avl from sorted list in linear time complexity, as presented in the class.
        if (start > end):
            return AVLTreeList()
        if end - start == 0:
            root = self.generate_new_node(lst[start])
            tree = AVLTreeList(root)
            tree.size = 1
            tree.update_firls_last()
            return tree
        smaller_tree = self.set_new_val_rec(lst, start, (end + start)// 2 - 1)
        bigger_tree = self.set_new_val_rec(lst, (end + start) // 2 + 1, end)
        median = lst[(end + start)//2]
        avl_median = AVLTreeList(self.generate_new_node(median))

        avl_median.size = 1
        avl_median.update_firls_last()
        if (smaller_tree.empty() and bigger_tree.empty()):
            return avl_median
        elif (smaller_tree.empty()):
            avl_median.root.right = bigger_tree.root
            bigger_tree.root.parent = avl_median.root
        elif (bigger_tree.empty()):
            avl_median.root.left = smaller_tree.root
            smaller_tree.root.parent = avl_median.root
        else:
            avl_median.root.right = bigger_tree.root
            bigger_tree.root.parent = avl_median.root
            avl_median.root.left = smaller_tree.root
            smaller_tree.root.parent = avl_median.root
        avl_median.root.update_node_fields()
        avl_median.size = avl_median.root.size
        avl_median.update_firls_last()
        return avl_median

    def generate_index(self, len):
        rand = random
        return rand.randint(0, len)

    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst): #O(max(log n,log m)) - m is the size of lst
        if (lst.empty() and not self.empty()):
            return self.root.height
        if (not lst.empty() and self.empty()):
            self.root = lst.root
            self.size = lst.size
            self.firstItem = lst.firstItem
            self.lastItem = lst.lastItem
            return lst.root.height
        if (self.empty() and lst.empty()):
            return 0
        if(self.root.height <= lst.root.height):
            connect_node = lst.retrieve_node(0)
            lst.delete(0)
        else:
            connect_node = self.retrieve_node(self.length()-1)
            self.delete(self.length()-1)
        h_diff = abs(lst.root.height - self.root.height)
        self_is_smaller = False
        if (self.length() <= lst.length()):
            h = self.root.height
            self_is_smaller = True
        else:
            h = lst.root.height
        if (h_diff == 0):  # even trees size
            connect_node.left = self.root
            self.root.parent = connect_node
            connect_node.right = lst.root
            lst.root.parent = connect_node
            self.root = connect_node
            self.root.update_node_fields()
            self.size = self.root.size
            self.update_firls_last()
        elif (self_is_smaller):
            curr_node = lst.root
            while curr_node.height > h:
                curr_node = curr_node.left
            parent = curr_node.parent
            connect_node.left = self.root
            self.root.parent = connect_node
            connect_node.right = curr_node
            curr_node.parent = connect_node
            parent.left = connect_node
            connect_node.parent = parent
            connect_node.update_node_fields()
            connect_node = parent
            while connect_node != None:
                connect_node.update_node_fields()
                if abs(connect_node.balance_factor) > 1:
                    connect_node, temp_cnt = self.rebalance_and_update(connect_node, 0)
                else:
                    connect_node.update_node_fields()
                prev = connect_node
                connect_node = connect_node.parent
            self.size = prev.size
            self.root = prev
            self.update_firls_last()
        else:
            connected_node = connect_node
            curr_node = self.root
            while curr_node.height > h:
                curr_node = curr_node.right
            parent = curr_node.parent
            connected_node.right = lst.root
            lst.root.parent = connected_node
            connected_node.left = curr_node
            curr_node.parent = connected_node
            connected_node.parent = parent
            parent.right = connected_node
            connected_node.update_node_fields()
            connected_node = parent
            while connected_node != None:
                connected_node.update_node_fields()
                if abs(connected_node.balance_factor) > 1:
                    connected_node, temp_cnt = self.rebalance_and_update(connected_node, 0)
                else:
                    connected_node.update_node_fields()
                prev = connected_node
                connected_node = connected_node.parent
            self.size = prev.size
            self.root = prev
            self.update_firls_last()
        return h_diff

    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val): #O(log n)
        if (self.empty()):
            return -1
        lst = self.listToArray()
        for i in range(len(lst)):
            if (lst[i] == val):
                return i
        return -1

    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root


    def append(self, val):
        return self.insert(self.length(), val)

    def update_firls_last(self):
        self.firstItem = self.retrieve_node(0)
        if (self.length() == 0):
            self.lastItem = self.firstItem
        else:
            self.lastItem = self.retrieve_node(self.length() - 1)

    def getTreeHeight(self):
        return self.root.height
