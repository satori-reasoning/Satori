from latex2sympy2 import latex2sympy, latex2latex


def transform_tex(tex: str, to: str) -> str:
    if to == "latex":
        return latex2latex(tex)
    elif to == "sympy":
        return latex2sympy(tex)
    else:
        raise ValueError(f"Unknown transformation target: {to}")


def check_tex_equiv(tex1: str, tex2: str) -> bool:
    for target in ["latex", "sympy"]:
        try:
            tex1_transformed = transform_tex(tex1, target)
            tex2_transformed = transform_tex(tex2, target)
            if tex1_transformed == tex2_transformed:
                return True
        except Exception as e:
            pass
    
    return None
