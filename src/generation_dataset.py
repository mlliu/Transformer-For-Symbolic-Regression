#can run right now, but still need more debug
# need a function to select expr
#process duaplicate expr
#check line by line, also compare with original one
"""
generation dataset
formula, datapoint(without random samples)
x1 datapoint , x2, ' '+formula y, formula
"""

import sympy as sp
import numpy as np
import re
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.cache import clear_cache
from sympy.integrals.risch import NonElementaryIntegral
from sympy.calculus.util import AccumBounds
from sympy import *
import csv

EXP_OPERATORS = {'exp', 'sinh', 'cosh'}
#SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '(', ')']

def count_nested_exp(s):
    """
    Return the maximum number of nested exponential functions in an infix expression.
    """
    stack = []
    count = 0
    max_count = 0
    for v in re.findall('[+-/*//()]|[a-zA-Z0-9]+', s):
        if v == '(':
            stack.append(v)
        elif v == ')':
            while True:
                x = stack.pop()
                if x in EXP_OPERATORS:
                    count -= 1
                if x == '(':
                    break
        else:
            stack.append(v)
            if v in EXP_OPERATORS:
                count += 1
                max_count = max(max_count, count)
    assert len(stack) == 0
    return max_count

def is_valid_expr(s):
    """
    Check that we are able to evaluate an expression (and that it will not blow in SymPy evaluation).
    """
    s = s.replace('Derivative(f(x),x)', '1')
    s = s.replace('Derivative(1,x)', '1')
    s = s.replace('(E)', '(exp(1))')
    s = s.replace('(I)', '(1)')
    s = s.replace('(pi)', '(1)')
    s = re.sub(r'(?<![a-z])(f|g|h|Abs|sign|ln|sin|cos|tan|sec|csc|cot|asin|acos|atan|asec|acsc|acot|tanh|sech|csch|coth|asinh|acosh|atanh|asech|acoth|acsch)\(', '(', s)
    count = count_nested_exp(s)
    if count >= 4:
        return False
    for v in EVAL_VALUES:
        try:
            local_dict = {s: (v + 1e-4 * i) for i, s in enumerate(EVAL_SYMBOLS)}
            value = ne.evaluate(s, local_dict=local_dict).item()
            if not (math.isnan(value) or math.isinf(value)):
                return True
        except (FloatingPointError, ZeroDivisionError, TypeError, MemoryError):
            continue
    return False

class GenerationDataset(object):
    #there is no sub,div, sqrt???
    SYMPY_OPERATORS = {
    # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        sp.exp: 'exp',
        sp.log: 'ln',
        sp.Abs: 'abs',
        sp.sign: 'sign',
        # Trigonometric Functions
        sp.sin: 'sin',
        sp.cos: 'cos',
        sp.tan: 'tan',
        sp.cot: 'cot',
        sp.sec: 'sec',
        sp.csc: 'csc',
        # Trigonometric Inverses
        sp.asin: 'asin',
        sp.acos: 'acos',
        sp.atan: 'atan',
        sp.acot: 'acot',
        sp.asec: 'asec',
        sp.acsc: 'acsc',
        # Hyperbolic Functions
        sp.sinh: 'sinh',
        sp.cosh: 'cosh',
        sp.tanh: 'tanh',
        sp.coth: 'coth',
        sp.sech: 'sech',
        sp.csch: 'csch',
         # Hyperbolic Inverses
        sp.asinh: 'asinh',
        sp.acosh: 'acosh',
        sp.atanh: 'atanh',
        sp.acoth: 'acoth',
        sp.asech: 'asech',
        sp.acsch: 'acsch',
    }
    
    #this operators are used to classify uniary and binary operator
    CLASSIFY_OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,
        'sign': 1,
        # Trigonometric Functions
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'cot': 1,
        'sec': 1,
        'csc': 1,
         # Trigonometric Inverses
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acot': 1,
        'asec': 1,
        'acsc': 1,
        # Hyperbolic Functions
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'coth': 1,
        'sech': 1,
        'csch': 1,
        # Hyperbolic Inverses
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acoth': 1,
        'asech': 1,
        'acsch': 1,
    } # no need for derivative and custom function
    #    # Derivative
    #    'derivative': 2,
    #    # custom functions
    #    'f': 1,
    #    'g': 2,
    #    'h': 3,
    #}
    
    def __init__(self):
        self.max_ops =5 #maximum number of operators at generation
        self.max_int =5 #max value of sampled integers
        #self.positive = True #sign of sampled integers
        self.max_len = 512#maximum length of generated equations
        self.leaf_probs = [0.5,0.25,0.25] # variables integers and constants
        self.positive = True # do not sample negative numbers
        self.int_base= 10
        self.balanced = False # in char_sp.py file
        self.classify_operators = sorted(list(self.CLASSIFY_OPERATORS.keys()))
        #operators and thier weight
        ops = "add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1"

        ops = ops.split(',')
        ops = sorted([x.split(':') for x in ops])

        self.all_ops = [o for o,_ in ops]
        self.una_ops = [o for o,_ in ops if self.CLASSIFY_OPERATORS[o]==1]
        self.bin_ops = [o for o,_ in ops if self.CLASSIFY_OPERATORS[o]==2]

        self.all_ops_probs = np.array([float(w) for _,w in ops]).astype(np.float64)
        self.una_ops_probs = np.array([float(w) for o,w in ops if self.CLASSIFY_OPERATORS[o]==1]).astype(np.float64)
        self.bin_ops_probs = np.array([float(w) for o,w in ops if self.CLASSIFY_OPERATORS[o]==2]).astype(np.float64)

        self.all_ops_probs =self.all_ops_probs/self.all_ops_probs.sum()
        self.una_ops_probs =self.una_ops_probs/self.una_ops_probs.sum()
        self.bin_ops_probs =self.bin_ops_probs/self.bin_ops_probs.sum()
        
        
        #constants
        #self.constants = ['pi','E']
        self.constants = ['pi']
        # variables, we only use x
        self.variables = {'x':sp.Symbol('x',real = True,nonzero=True)}
        self.n_variables =len(self.variables)
        #no coefficients, no functions,no symbols, no balanced 

        #no, we have symbols in write_int may be need to delete some one
        #self.symbols = ['I', 'INT+', 'INT-', 'INT','FLOAT', '-', '.', '10^', 'Y', "Y'", "Y''"]
        #if self.balanced:
        #    assert self.int_base > 2
        #    max_digit = (self.int_base + 1) // 2
        #    self.elements = [str(i) for i in range(max_digit - abs(self.int_base), max_digit)]
        #else:
        #    self.elements = [str(i) for i in range(abs(self.int_base))]

        #vocabulary combination of all symbols
        #self.words = SPECIAL_WORDS + self.constants +list(self.variables.keys())+self.classify_operators+self.symbols+self.elements #where is integer
        #self.id2word = {id: w for id,w in enumerate(self.words)}
        #self.wordid = {w:id for id,w in enumerate(self.words)}

        #number of words
        #self.n_words = len(self.words)
        #self.eos_index = 0
        #self.pad_index = 1

        #leaf probability [0.5,0.25,0.25] # variables integers and constants
        self.leaf_probs = np.array(self.leaf_probs).astype(np.float64)
        self.leaf_probs = self.leaf_probs/self.leaf_probs.sum()
        #number of leaves
        self.n_leaves = len(self.variables) + self.max_int *(1 if self.positive else 2)+ len(self.constants)


        # generation parameters
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)
        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(self.max_ops)
        self.ubi_dist = self.generate_ubi_dist(self.max_ops)
     
    
    def generate_bin_dist(self, max_ops):
            """
            `max_ops`: maximum number of operators
            Enumerate the number of possible binary trees that can be generated from empty nodes.
            D[e][n] represents the number of different binary trees with n nodes that
            can be generated from e empty nodes, using the following recursion:
                D(0, n) = 0
                D(1, n) = C_n (n-th Catalan number)
                D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
            """
            # initialize Catalan numbers
            catalans = [1]
            for i in range(1, 2 * max_ops + 1):
                catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

            # enumerate possible trees
            D = []
            for e in range(max_ops + 1):  # number of empty nodes
                s = []
                for n in range(2 * max_ops - e + 1):  # number of operators
                    if e == 0:
                        s.append(0)
                    elif e == 1:
                        s.append(catalans[n])
                    else:
                        s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
                D.append(s)
            return D
    
    def generate_ubi_dist(self, max_ops):
            """
            `max_ops`: maximum number of operators
            Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
            D[e][n] represents the number of different binary trees with n nodes that
            can be generated from e empty nodes, using the following recursion:
                D(0, n) = 0
                D(e, 0) = L ** e
                D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
            """
            # enumerate possible trees
            # first generate the tranposed version of D, then transpose it
            D = []
            D.append([0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)]))
            for n in range(1, 2 * max_ops + 1):  # number of operators
                s = [0]
                for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                    s.append(self.nl * s[e - 1] + self.p1 * D[n - 1][e] + self.p2 * D[n - 1][e + 1])
                D.append(s)
            assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
            D = [[D[j][i] for j in range(len(D)) if i < len(D[j])] for i in range(max(len(x) for x in D))]
            return D
    
    
    
    
    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1])
        for i in range(nb_empty):
            probs.append((self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1])
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
                                             #compute the ubi_dist is to generate four expression trees with same probability
        e = rng.choice(2 * nb_empty, p=probs)# random select a number from range(2*nb_empty) with p = probs
        arity = 1 if e < nb_empty else 2     #1 is for uni 2 is for binary
        e = e % nb_empty                     #e is skipped, is the position of the operator
        return e, arity
    
    #101 to ['INT+', '1', '0', '1']
    def write_int(self, val):  
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        base = self.int_base
        balanced = False#self.balanced
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if base < 0 or balanced:
            res.append('INT')
        else:
            res.append('INT-' if neg else 'INT+')
        return res[::-1]
    
    def get_leaf(self, max_int, rng):
        """
        leaf probability [0.5,0.25,0.25] # variables integers and constants
        Generate a leaf ramdomly follow the leaf probabiliy.
        """
        self.leaf_probs
        leaf_type = rng.choice(3, p=self.leaf_probs)
        if leaf_type == 0:
            return [list(self.variables.keys())[rng.randint(len(self.variables))]]
        elif leaf_type == 1:
            c = rng.randint(1, max_int + 1)
            c = c if (self.positive or rng.randint(2) == 0) else -c
            return self.write_int(c)
        else:
            return [self.constants[rng.randint(len(self.constants))]]
     
    def _generate_expr(self, nb_total_ops, max_int, rng, require_x=False, require_y=False, require_z=False):
        """
        Create a tree with exactly `nb_total_ops` operators.
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += self.CLASSIFY_OPERATORS[op] - 1 - skipped  # created empty nodes - skipped future leaves
            t_leaves += self.CLASSIFY_OPERATORS[op] - 1            # update number of total leaves
            l_leaves += skipped                           # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = stack[:pos] + [op] + [None for _ in range(self.CLASSIFY_OPERATORS[op])] + stack[pos + 1:]
         # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        # create leaves
        # optionally add variables x, y, z if possible
        assert not require_z or require_y
        assert not require_y or require_x
        leaves = [self.get_leaf(max_int, rng) for _ in range(t_leaves)]
        if require_z and t_leaves >= 2:
            leaves[1] = ['z']
        if require_y:
            leaves[0] = ['y']
        if require_x and not any(len(leaf) == 1 and leaf[0] == 'x' for leaf in leaves):
            leaves[-1] = ['x']
        rng.shuffle(leaves)

        # insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1:]
        assert len(leaves) == 0

        return stack
    
    def infix_to_sympy(self, infix, no_rewrite=False):
        """
        Convert an infix expression to SymPy.
        """
        #deny this function temporarily
        #if not is_valid_expr(infix):
        #    raise ValueErrorExpression
        #it seems there is no need for local_dict
        #expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        expr = parse_expr(infix, evaluate=True)
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression
        #deny this function temporarily
        #if not no_rewrite:
        #    expr = self.rewrite_sympy_expr(expr)
        return expr
    
    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'rac':
            return f'({args[0]})**(1/({args[1]}))'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'pow2':
            return f'({args[0]})**2'
        elif token == 'pow3':
            return f'({args[0]})**3'
        elif token == 'pow4':
            return f'({args[0]})**4'
        elif token == 'pow5':
            return f'({args[0]})**5'
        elif token in ['sign', 'sqrt', 'exp', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']:
            return f'{token}({args[0]})'
        elif token == 'derivative':
            return f'Derivative({args[0]},{args[1]})'
        elif token == 'f':
            return f'f({args[0]})'
        elif token == 'g':
            return f'g({args[0]},{args[1]})'
        elif token == 'h':
            return f'h({args[0]},{args[1]},{args[2]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token
        raise InvalidPrefixExpression(f"Unknown token in prefix expression: {token}, with arguments {args}")

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        balanced = self.balanced
        val = 0
        if not (balanced and lst[0] == 'INT' or base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
            raise InvalidPrefixExpression(f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1
    
    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.classify_operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.CLASSIFY_OPERATORS[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        #no coefficients
        #elif t in self.variables or t in self.coefficients or t in self.constants or t == 'I':
        elif t in self.variables or t in self.constants or t == 'I':
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]
        
    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'
    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # derivative operator
        """
        if op == 'derivative':
            assert n_args >= 2
            assert all(len(arg) == 2 and str(arg[0]) in self.variables and int(arg[1]) >= 1 for arg in expr.args[1:]), expr.args
            parse_list = self.sympy_to_prefix(expr.args[0])
            for var, degree in expr.args[1:]:
                parse_list = ['derivative' for _ in range(int(degree))] + parse_list + [str(var) for _ in range(int(degree))]
            return parse_list
        """
        assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # square root
        if op == 'pow' and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ['sqrt'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list
    
    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ['div'] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']
        elif expr == sp.I:
            return ['I']
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # environment function
        #for func_name, func in self.functions.items():
        #    if isinstance(expr, func):
       #         return self._sympy_to_prefix(func_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")
        
    def gen_func_points(self,rng,datalength):
        """
        Generate pairs of (function, datapoints)
        start by generating a random function f, and use SymPy to generate datapoints
        """
        #seed = 1 #1 for random seed
        #rng = np.random.RandomState(seed)
        
        #x = self.variables['x']
        x = sp.Symbol('x')
        
        if rng.randint(40) ==0:   #randint (0,40). the probability for true is 1/40 
            nb_ops = rng.randint(0,3) # generate a random number from (0,3)
        else:
            nb_ops = rng.randint(3,self.max_ops+1) #(3,16) #the total number of ops
        
        self.stats = np.zeros(10,dtype = np.int64)
        
        print(nb_ops)
        #try:
            #generate an expression and rewrite it
            #avoid issues in 0 and convert to SymPy
        f_expr = self._generate_expr(nb_ops,self.max_int,rng)
        infix = self.prefix_to_infix(f_expr)
        f = self.infix_to_sympy(infix)
         # skip constant expressions
        if x not in f.free_symbols:
            return None,None,None
        #no need for this procedure, because there is no coefficients
        # remove additive constant, re-index coefficients
        #if rng.randint(2) == 0:
        #    f = remove_root_constant_terms(f, x, 'add') # this can be removed
        #f = self.reduce_coefficients(f)
        #f = self.simplify_const_with_coeff(f)
        #f = self.reindex_coefficients(f)

        # generate dataset
        function = lambdify(x, f)
        data_x = np.linspace(-10,10,datalength)
        data_y = function(data_x)

        # write (data_x,data_y) and f_prefix 
        #data_x and data_y is used as input of model maybe csv
        #prefix is used to model training as output
        f_prefix = self.sympy_to_prefix(f)
        #except TimeoutError:
        #    raise
        #except (ValueError, AttributeError, TypeError, OverflowError, NotImplementedError, UnknownSymPyOperator, ValueErrorExpression):
        #    return None
        #except Exception as e:
        #    logger.error("An unknown exception of type {0} occurred in line {1} for expression \"{2}\". Arguments:{3!r}.".format(type(e).__name__, sys.exc_info()[-1].tb_lineno, infix, e.args))
        #    return None
        
        #return f_prefix
        return f_prefix, data_x, data_y
    

if __name__ == "__main__":
    
    seed = 1 #1 for random seed
    rng = np.random.RandomState(seed)
    
    datalength = 512 # n_embd=8, T=64  
    
    #path for dataset
    point_path ='../data/datapoint_2.tsv'
    expr_pth = '../data/expr_2.tsv'
    #generate dataset
    datasize=10
    #init the data generator
    generator = GenerationDataset()
    
    for i in range(datasize):
        
        
            #generate points and funcs
            #f_prefix = generator.gen_func_points(rng)
            f_prefix, data_x,data_y = generator.gen_func_points(rng,datalength)
            if f_prefix is None:
                continue
            print(f_prefix)
            print(len(data_y))
            
            # self.file_handler_prefix = io.open(params.export_path_prefix, mode='a', encoding='utf-8')
            #f.file_handler_prefix.write(f'{prefix1_str}\t{prefix2_str}\n')
            with open(expr_pth,'a') as expr_file:
                expr_writer = csv.writer(expr_file, delimiter='\t')
                expr_writer.writerow(f_prefix)
            
            with open(point_path,'a') as dp_file:
                dp_writer = csv.writer(dp_file, delimiter='\t')
                dp_writer.writerow(data_y)

        
