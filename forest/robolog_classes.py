from typing import Tuple, List


class Inequality:

    def __init__(self, feature: int, op: str, val: float):
        self.op_map = {'gt': '>', 'leq': '<='}
        self.op = op
        self.feature = feature
        self.val = val

    def __repr__(self):
        return ' '.join(['s' + str(self.feature), self.op_map[self.op], str(self.val)])


# class Variable:
#
#     def __init__(self, name: str, type: str):
#         self.name = name
#         self.type = type
#
#     def __repr__(self):
#         return self.name + ': ' + self.type

class Predicate:
    def __init__(self, name, variables):
        self.name = name
        self.variables = variables

    def __repr__(self):
        return self.name + '(' + ', '.join(self.variables) + ')'


class Literal:
    def __init__(self, name: str, variables: List[str]):
        self.name = name
        self.variables = variables

    def __repr__(self):
        variable_string = '(' + ', '.join([v.__repr__() for v in variables]) + ')'
        return self.name + variable_string


class Rule:
    # body is a list of Predicates and Inequalities
    def __init__(self, head: Predicate, body: list):
        self.head = head
        self.body = body

    def __repr__(self):
        inequalities = [pred.__repr__() for pred in self.body if type(pred) == Inequality]
        predicates = [pred.__repr__() for pred in self.body if type(pred) == Predicate]
        body_str = ', '.join(predicates + inequalities)
        rule_str = self.head.__repr__() + " :- " + body_str
        return rule_str
