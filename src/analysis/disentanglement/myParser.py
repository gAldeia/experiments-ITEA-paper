import pandas as pd
import numpy  as np

import re

from scipy import stats

from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
)

# Sympy is not safe for parsing.
# https://docs.sympy.org/latest/modules/core.html#module-sympy.core.sympify

# We'll make a parser based on the example provided here:
#https://github.com/pyparsing/pyparsing/blob/master/examples/fourFn.py


epsilon = 1e-12

# Operations with infix notation
opn = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    "^": np.power,
    
    "<" : lambda x, y: 1.0 if x < y else 0.0,
    "leq" : lambda x, y: 1.0 if x <= y else 0.0,
    ">" : lambda x, y: 1.0 if x > y else 0.0,
    "geq" : lambda x, y: 1.0 if x >= y else 0.0,
    'XOR': lambda x, y: (x and not y) or (not x and y),
    'OR': lambda x, y: 1.0 if (x or y) else 0,
    'AND': lambda x, y: 1.0 if (x and y) else 0.0,
    'eq' : lambda x, y: 1.0 if x==y else 0.0
}

# functions
fn = {
    "sin"     : np.sin,
    "cos"     : np.cos,
    "exp"     : np.exp,
    "tanh"    : np.tanh,
    "id"      : lambda x: x,
    "log"     : np.log,
    "SQRTABS" : lambda x: np.sqrt(np.abs(x)),

    # FEAT FULL
    "relu"    : lambda x: np.ma.array(x, mask=(x<=0.0), fill_value=0).filled(),
    "gauss"   : lambda x: np.exp(-np.power(x, 2)),    
    "logit"   : lambda x: np.ma.array(np.log(x/(1-x)), mask=(x==0.0), fill_value=0).filled(),
    "NOT"     : lambda x: 0.0 if x != 0.0 else 1.0,
    "if"      : lambda x, y, z: y if x!= 0.0 else z,
    "float"   : float,
    
    # https://lacava.github.io/feat/cpp_api/d1/dcb/n__gaussian_8cc_source.html
}



exprStack = []

def push_first(toks):
    exprStack.append(toks[0])

def push_unary_minus(toks):
    for t in toks:
        if t == "-":
            exprStack.append("unary -")
        else:
            break

bnf = None

def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    
    if not bnf:
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")
        
        # fnumber = Combine(Word("+-"+nums, nums) +
        #                    Optional("." + Optional(Word(nums))) +
        #                    Optional(e + Word("+-"+nums, nums)))
        # or use provided pyparsing_common.number, but convert back to str:
        # fnumber = ppc.number().addParseAction(lambda t: str(t[0]))
        
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")

        # Basic Operations and logical comparators 
        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        les, leq, ge, geq, xor, eq, or_, and_ = Literal('<'), Literal('leq'), Literal('>'), Literal('geq'), Literal('XOR'), Literal('eq'), Literal('OR'), Literal('AND')
        
        # grouping the operators
        addop  = plus | minus
        multop = mult | div
        expop  = Literal("^")
        logic = les | leq | ge | geq | xor | eq | or_ | and_

        #Forward declaration of an expression to be
        #defined later - used for recursive grammars, such as algebraic infix notation.
        expr_logic = Forward()
        factor     = Forward()
        expr       = Forward()
        
        expr_list = delimitedList(Group(expr_logic))
        
        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))

        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            insert_fn_argcount_tuple
        )
        atom = (
            logic[...]
            + (
                (fn_call | pi | e | fnumber | ident).setParseAction(push_first)
                | Group(lpar + expr_logic + rpar)
            )
        ).setParseAction(push_unary_minus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        
        # precedence: power
        factor <<= atom + (expop + factor).setParseAction(push_first)[...]
        
        # precedence: multiplication, sum
        term = factor + (multop + factor).setParseAction(push_first)[...]
        
        expr <<= term + (addop + term).setParseAction(push_first)[...]
        
        # precedence: logical operations
        expr_logic <<= expr + (logic + expr).setParseAction(push_first)[...]
        
        bnf = expr_logic
        
    return bnf

def evaluate_stack(s, xs):
    op, num_args = s.pop(), 0
    
    if isinstance(op, tuple):
        op, num_args = op
    
    if op == "unary -":
        return -evaluate_stack(s, xs)
    
    if op in [r'+', r'-', r'*', r'/', r'^', r'<', r'leq', r'>', r'geq', r'XOR', 'eq', 'OR', 'AND']:
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s, xs)
        op1 = evaluate_stack(s, xs)

        aux =  opn[op](op1, op2)
    
        # creating boundaries on the operators
        if np.isnan(aux):
            return 0.0
        elif (aux > 1e+150) or np.isposinf(aux):
            return 1e+200
        elif (aux < -1e+150) or np.isneginf(aux):
            return -1e+200
        else:
            return aux
        
    elif op == "PI":
        return math.pi  # 3.1415926535
    
    elif op == "E":
        return math.e  # 2.718281828
    
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([evaluate_stack(s, xs) for _ in range(num_args)])
        aux  = fn[op](*args)
    
        # same boundaries, for the functions now
        if np.isnan(aux):
            return 0.0
        elif (aux > 1e+150) or np.isposinf(aux):
            return 1e+200
        elif (aux < -1e+150) or np.isneginf(aux):
            return -1e+200
        else:
            return aux
        
    elif "x" == op[0]:
        
        # Handling the occurence of variables by
        # getting the index of the variable and
        # returning the value from the given sample xs
        idx = int(op.replace('x_', ''))
        
        return xs[idx]
    
    elif op[0].isalpha():
        raise Exception("invalid identifier '%s'" % op)
    
    else:
        # try to evaluate as int first, then as float if int fails
        try:
            return int(op)
        
        except ValueError:
            return float(op)
        


def test(s, expected, xs):
    exprStack[:] = []
    
    try:
        results = BNF().parseString(s, parseAll=True)
        val = evaluate_stack(exprStack[:], xs)
        
    except ParseException as pe:
        print(s, "failed parse:", str(pe))
        
        return 0.0
    
    except Exception as e:
        print(s, "failed eval:", str(e), exprStack)
        
        return 0.0
    else:
        if val == expected:
            print(s, "=", val, results, "=>", exprStack)
        else:
            print(s + "!!!", val, "!=", expected, results, "=>", exprStack)   
        return val
    

# Parser
def parse(s, X):
    exprStack[:] = []
    
    val = []
    try:
        
        results = BNF().parseString(s, parseAll=True)
        
    except ParseException as pe:
        print("ERROR ON BNF")
        print(s, "failed parse:", str(pe))
    except Exception as e:
        print("ERROR ON BNF")
        print(s, "failed eval:", str(e), exprStack)

    for xs in X:
        try:
            aux = evaluate_stack(exprStack[:], xs)

        except ParseException as pe:
            #print(s, "failed parse:", str(pe))
            print("failed parse")

            val.append(0.0)

        except Exception as e:
            print(s, "failed eval:", str(e), exprStack)
            #print("failed eval", str(e))

            val.append(0.0)

        else:
            if np.isnan(aux):
                print('Eval is returning NaN')

                val.append(0.0)

            elif np.isinf(aux):
                print('Eval is returning Inf')

                val.append(np.exp(300))

            else:
                val.append(aux)
    return np.array(val)