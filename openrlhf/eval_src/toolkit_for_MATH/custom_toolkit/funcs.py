import sympy


def parse_function(expr):
    """
    Parse a function definition of the form "f(x) = expression"
    and return a symbolic function representation.
    """
    # Remove spaces
    expr = expr.replace(" ", "")
    
    # Remove dollar signs
    expr = expr.replace("$", "")
    
    # Split function name and body
    func_part, body = expr.split("=")
    
    # Extract function variable (e.g., "f(z)" -> "z")
    var_name = func_part[func_part.find("(")+1 : func_part.find(")")]
    
    # Define symbolic variable
    var = sympy.Symbol(var_name)
    
    # Convert the function body to a symbolic expression
    func_expr = sympy.sympify(body, locals={var_name: var})
    
    return var, func_expr


def check_funcs_equiv(expr1, expr2):
    """
    Check if two function definitions are equivalent by normalizing them.
    """
    try:
        var1, func1 = parse_function(expr1)
        var2, func2 = parse_function(expr2)
        
        # Rename variables to a common name (e.g., 'x') and compare
        x = sympy.Symbol('x')
        func1 = func1.subs(var1, x)
        func2 = func2.subs(var2, x)

        return sympy.simplify(func1 - func2) == 0  # True if functions are equivalent
    except Exception as e:
        pass
    
    return None
