"""
This module defines export functions for decision trees.
"""

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
# Licence: BSD 3 clause

from warnings import warn

# from sklearn.externals import six

from sklearn.tree import _tree


def export_graphviz(decision_tree, out_file="tree.dot", feature_names=None,
                    max_depth=None, close=None):
    """Export a decision tree in DOT format.
    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::
        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)
    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.
    out_file : file object or string, optional (default="tree.dot")
        Handle or name of the output file.
    feature_names : list of strings, optional (default=None)
        Names of each of the features.
    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()
    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf,
    ...     out_file='tree.dot')                # doctest: +SKIP
    """
    if close is not None:
        warn("The close parameter is deprecated as of version 0.14 "
             "and will be removed in 0.16.", DeprecationWarning)

    def node_to_str(tree, node_id, criterion):
        if not isinstance(criterion, six.string_types):
            criterion = "impurity"

        value = tree.value[node_id]
        if tree.n_outputs == 1:
            value = value[0, :]

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return "%s = %.4f\\nsamples = %s\\nvalue = %s" \
                   % (criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id],
                      value)
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X[%s]" % tree.feature[node_id]

            return "%s <= %.4f\\n%s = %s\\nsamples = %s" \
                   % (feature,
                      tree.threshold[node_id],
                      criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id])


    def recurse(tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if max_depth is None or depth <= max_depth:
            out_file.write('%d [label="%s", shape="box"] ;\n' %
                           (node_id, node_to_str(tree, node_id, criterion)))

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)

        else:
            out_file.write('%d [label="(...)", shape="box"] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

    own_file = False
    if isinstance(out_file, six.string_types):
        if six.PY3:
            out_file = open(out_file, "w", encoding="utf-8")
        else:
            out_file = open(out_file, "wb")
        own_file = True

    out_file.write("digraph Tree {\n")

    if isinstance(decision_tree, _tree.Tree):
        recurse(decision_tree, 0, criterion="impurity")
    else:
        recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)
    out_file.write("}")

    if own_file:
        out_file.close()

def export_dict(tree, feature_names=None, max_depth=None) :
    """Export a decision tree in dict format.
    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported
    feature_names : list of strings, optional (default=None)
        Names of each of the features.
    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.
    Returns
    -------
    a dictionary of the format <tree> := {
        'feature' <int> | <string>,
        'threshold' : <float>,
        'impurity' : <float>,
        'n_node_samples' : <int>,
        'left' : <tree>,
        'right' : <tree>,
        'value' : [<int>],
    }
    if feature_names is provided, it is used to map feature indicies
    to feature names.  All types (including the value list) are native
    python types as opposed to numpy types to make exporting to json
    and other pythonic operations easier.
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> import json
    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()
    >>> clf = clf.fit(iris.data, iris.target)
    >>> d = tree.export_dict(clf)
    >>> j = json.dumps(d, indent=4)
    """
    tree_ = tree.tree_

    # i is the element in the tree_ to create a dict for
    def recur(i, depth=0) :
        if max_depth is not None and depth > max_depth :
            return None
        if i == _tree.TREE_LEAF :
            return None

        feature = int(tree_.feature[i])
        threshold = float(tree_.threshold[i])
        if feature == _tree.TREE_UNDEFINED :
            feature = None
            threshold = None
            value = [map(int, l) for l in tree_.value[i].tolist()]
            value = [list(m) for m in value]
        else :
            value = None
            if feature_names :
                feature = feature_names[feature]
        # if value is not None:
           # print(value)
        return {
            'feature' : feature,
            'threshold' : threshold,
            #'impurity' : float(tree_.impurity[i]),
            #'n_node_samples' : int(tree_.n_node_samples[i]),
            'left'  : recur(tree_.children_left[i],  depth + 1),
            'right' : recur(tree_.children_right[i], depth + 1),
            'value' : value
        }

    return recur(0)