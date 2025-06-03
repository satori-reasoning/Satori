# ---------------------------------------------------------
# Xwin-Math
# Copyright (c) 2023 Xwin-Math Team
# Licensed under The MIT License [see LICENSE for details]
# Based on ToRA (https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py)
# Modified by Weiqi Wang
# ---------------------------------------------------------
import re
from typing import Union, Any
from copy import deepcopy
from math import isclose
from sympy import simplify, sympify, N
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from itertools import permutations

from .parsing_lib import *
# from .custom_toolkit.latex import transform_tex
# from src.evaluation.utils.custom_toolkit.latex import check_tex_equiv
# from src.evaluation.utils.custom_toolkit.funcs import check_funcs_equiv


def has_numbers(input_string: str) -> bool:
    """
    Checks if a string contains a number.
    """
    return any(char.isdigit() for char in input_string)


def has_structure(input_string: str) -> bool:
    """
    Checks if a string contains structured content.
    """
    if "(" in input_string or ")" in input_string or "[" in input_string or "]" in input_string or "\\" in input_string or "<" in input_string or ">" in input_string or "," in input_string or 'x' in input_string or 'y' in input_string or 'z' in input_string:
        return True
    return False


def sympy_parse(input_string: str) -> Any:
    """
    Parsing strings into mathematical expressions using sympy
    """
    for f in [parse_latex, parse_expr]:
        #!!!!!!! IMPORTANT: This is a temperary fix for sympy parsing issue
        if f == parse_expr:
            if "^" in input_string:
                input_string = input_string.replace("^", "**")
        
        try:
            return f(input_string)
        except:
            pass
        
    return input_string


def symbolic_equal(a: str, b: str) -> Union[bool, None]:
    """
    Check if two strings are symbolic equal.
    """
    a = sympy_parse(a)
    b = sympy_parse(b)
    
    try:
        expr1 = a.replace(")(", ")*(").replace("^", "**")
        expr2 = b.replace(")(", ")*(").replace("^", "**")
        simplified_expr1 = simplify(sympify(expr1))
        simplified_expr2 = simplify(sympify(expr2))
        if simplified_expr1 == simplified_expr2:
            return True
    except:
        pass

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), float(N(a)), rel_tol=1e-9) and isclose(N(a), float(N(a)), rel_tol=1e-9):
            return False
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    
    return None


def convert_to_int(input_string: str) -> Union[int, None]:
    """
    Try to convert a string into int. Return `None` if an error occurs.
    """
    try:
        float_s = float(input_string)
        int_s = int(float_s)

        # If a floating-point number is converted to an integer that is very close to itself, then we consider it to be an integer.
        if isclose(int_s, float_s, rel_tol=1e-9):
            return int_s
        
        return None
    except:
        return None
    

def frac_to_float(frac_str):
    """
    Convert a LaTeX \\frac{}{} expression to a float.

    Parameters:
    frac_str (str): The \\frac{}{} string to convert.

    Returns:
    float: The numerical value of the fraction.
    """
    assert frac_str.startswith("\\frac{") and "}{" in frac_str and frac_str.endswith("}")
    # Regular expression to match \frac{numerator}{denominator}
    match = re.match(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}", frac_str)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        return numerator / denominator
    
    raise ValueError("Input is not a valid \\frac{}{} string")


def convert_to_float(input_string: str) -> Union[float, None]:
    """
    Try to convert a string into float. Return `None` if an error occurs.
    """
    try:
        return float(input_string)
    except:
        pass
    
    return None


def convert_base_n_to_decimal(input_string: str):
    """
    Convert a number expressed in base-n format (a_b or a_{b}) to decimal.

    Parameters:
    input_string (str): The number in base-n format.

    Returns:
    int: The decimal equivalent of the number.
    """
    match = re.match(r"(\d+)_\{?(\d+)\}?", input_string)
    if match:
        value = match.group(1)
        base = match.group(2)
        return int(value, int(base))

    return None


def sqrt_to_float(sqrt_str):
    """
    Convert a LaTeX \\sqrt[]{} expression to a float.
    
    Parameters:
    sqrt_str (str): The \\sqrt[]{} string to convert.
    
    Returns:
    float: The numerical value of the square root.
    """
    assert sqrt_str.startswith("\\sqrt") and sqrt_str.endswith("}")
    # Regular expression to match \sqrt[degree]{radicand}
    match = re.match(r"^\\sqrt(?:\[(\d+)\])?\{(-?\d+)\}", sqrt_str)
    if match:
        degree = int(match.group(1)) if match.group(1) else 2
        radicand = int(match.group(2))
        return radicand ** (1 / degree)
    
    raise ValueError("Input is not a valid \sqrt[]{} string")


#TODO: this one does not work when running RL, don't know why
# def convert_latex_numerical_string_to_float(latex_str):
#     """
#     Evaluate a LaTeX string to its numerical value.
#     Handles nested fractions and square roots.
#     """
#     try:
#         res = transform_tex(latex_str, "sympy")
#         res = float(res)
#         return res
#     except:
#         pass
    
#     return None


def convert_latex_numerical_string_to_float(latex_str):
    """
    Evaluate a LaTeX string to its numerical value.
    Handles nested fractions and square roots.
    """
    # Remove outer parentheses if they exist
    #! Order sensitive!
    possible_parentheses = [
        ("\\left(", "\\right)"),
        ("(", ")"),
        ("\\left[", "\\right]"),
        ("[", "]"),
        ("\\left\\{", "\\right\\}"),
        ("{", "}"),
    ]
    for open_paren, close_paren in possible_parentheses:
        if latex_str.startswith(open_paren) and latex_str.endswith(close_paren):
            latex_str = latex_str[len(open_paren):-len(close_paren)]
            latex_str = latex_str.strip()
            break

    # Handle plain numbers
    try:
        return float(latex_str)
    except:
        pass
    
    # Handle fractions
    try:
        return frac_to_float(latex_str)
    except:
        pass
    
    # Handle square roots
    try:
        return sqrt_to_float(latex_str)
    except:
        pass

    return None


def convert_percent_to_decimal(text):
    """
    Converts all occurrences of 'xxx%' in a string to their decimal format.
    """
    assert isinstance(text, str)
    assert "%" in text
    
    def percentage_to_decimal(match):
        percent_value = float(match.group(1)) / 100  # Convert to decimal
        return str(percent_value)  # Replace with decimal format

    # Find and replace all percentages
    result = re.sub(r"(\d+\.?\d*)%", percentage_to_decimal, text)
    return result


def numerical_equal(a: str, b: str) -> Union[bool, None]:
    """
    Check if two strings are numerical equal.
    """
    # Try integer comparison
    a_int = convert_to_int(a)
    b_int = convert_to_int(b)

    if a_int is not None and b_int is not None:
        return a_int == b_int

    # Try float comparison
    a_float = convert_to_float(a)
    b_float = convert_to_float(b)

    if a_float is not None and b_float is not None:
        return isclose(a_float, b_float, rel_tol=1e-3)
    
    # Try different number systems
    a_decimal = convert_base_n_to_decimal(a)
    b_decimal = convert_base_n_to_decimal(b)
    
    if a_decimal is not None and b_decimal is not None:
        return a_decimal == b_decimal
    
    # Try deconstruct latex expressions
    value_a = convert_latex_numerical_string_to_float(a)
    value_b = convert_latex_numerical_string_to_float(b)
    
    if value_a is not None and value_b is not None:
        return isclose(value_a, value_b, rel_tol=1e-3)

    return None


def literal_check(model_generated_answer: str, ground_truth: str) -> Union[bool, None]:
    """
    Check if two strings are the same character by character
    """
    assert isinstance(model_generated_answer, str)
    assert isinstance(ground_truth, str)
    
    model_remove = deepcopy(model_generated_answer).replace(",", " ").replace(" ", "").replace(" ", "")
    gt_remove = deepcopy(ground_truth).replace(",", " ").replace(" ", "").replace(" ", "")
    if model_remove == gt_remove:
        return True
    
    for model_ans, gt_ans in permutations([model_remove, gt_remove]):
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]
        for pair in pairs:
            left, right = pair[0], pair[1]
            if model_ans.startswith(left) and model_ans.endswith(right):
                if gt_ans in model_ans:
                    return True

    if has_numbers(model_generated_answer) is False and has_numbers(ground_truth) is False:
        model_generated_answer = model_remove.strip("[]() ")
        ground_truth = gt_remove.strip("[]() ")
        if model_generated_answer == ground_truth:
            return True

    return False


def number_check(model_generated_answer: str, ground_truth: str) -> None:
    """
    Check if two strings have the same mathematical meaning.
    """
    if "," in model_generated_answer or "," in ground_truth:
        return None

    model_generated_answer = remove_prefix_and_suffix(remove_equals(model_generated_answer))
    ground_truth = remove_prefix_and_suffix(remove_equals(ground_truth))

    numerical_equal_result = numerical_equal(model_generated_answer, ground_truth)
    if numerical_equal_result is not None:
        return numerical_equal_result

    symbolic_equal_result = symbolic_equal(model_generated_answer, ground_truth)

    if symbolic_equal_result is not None:
        return symbolic_equal_result

    return None


def _latex_equiv(model_answer: str, gt_answer: str) -> bool:
    assert gt_answer is not None
    assert isinstance(gt_answer, str)
    assert len(gt_answer) > 0

    if model_answer is None or not isinstance(model_answer, str) or len(model_answer) == 0:
        return False

    model_answer = string_normalization(model_answer)
    gt_answer = string_normalization(gt_answer)
    
    #! First try: literal check
    # Compare strings character by character after simple processing including remove $%.
    # First we remove the boxes in the string but keeps the content
    # \text{apple} --> apple
    model_ans_norm_wo_boxes = remove_boxes_keep_content(model_answer)
    model_ans_norm_wo_boxes_wo_prefix_suffix = remove_prefix_and_suffix(model_ans_norm_wo_boxes)
    gt_norm_wo_boxes = remove_boxes_keep_content(gt_answer)
    gt_norm_wo_boxes_wo_prefix_suffix = remove_prefix_and_suffix(gt_norm_wo_boxes)

    literal_check_result = literal_check(model_ans_norm_wo_boxes_wo_prefix_suffix, gt_norm_wo_boxes_wo_prefix_suffix)
    if literal_check_result is True:
        return True

    #! Second try: number check
    # Treat a string as a single number/extract a single number from a string and then compare.
    # If we can accept a few mistakes, we try to extract numbers from the answers and compare them
    
    # Handle percentages
    if "%" in model_answer or "%" in gt_answer:
        if "%" in model_answer:
            model_answer_no_percent = convert_percent_to_decimal(model_answer)
        else:
            model_answer_no_percent = model_answer
        if "%" in gt_answer:
            gt_answer_no_percent = convert_percent_to_decimal(gt_answer)
        else:
            gt_answer_no_percent = gt_answer
        
        if any(number_check(x, y) for x in [model_answer_no_percent, model_answer] for y in [gt_answer_no_percent, gt_answer]):
            return True
    
    # We wan't to use raw model_answer to keep the $$
    # $13$ meters --> $13$ --> 13
    model_ans_num_lst = search_for_numbers(model_answer)

    # We want the original answer has $$
    # This way we are able to consider the answer as a whole
    # We don't want \frac{13}{4} --> [13, 4] to be considered as 2 numbers
    if gt_answer[0] != "$":
        gt_answer = "$" + gt_answer
    if gt_answer[-1] != "$":
        gt_answer = gt_answer + "$"
    gt_num_lst = search_for_numbers(gt_answer)

    # We want to judge only those answers that contain only one number that represents the full meaning of the original string.
    # If the string still has LaTeX components or variables in addition to this number, then we believe that this number may not represent the meaning of the answer.
    # Here we must be really really careful.
    # x \\leq -5 vs. x \\geq -5
    # (-\\infty, 5) vs. (5, +\\infty)
    # TODO: We may have better methods to check if the numbers are simple enough
    if len(model_ans_num_lst) == 1 and len(gt_num_lst) == 1 and \
        not has_structure(model_answer.replace(model_ans_num_lst[0], "")) and \
        not has_structure(gt_answer.replace(gt_num_lst[0], "")):

        model_num = remove_prefix_and_suffix(remove_boxes_keep_content(remove_text_box_only(model_ans_num_lst[0])))
        gt_num = remove_prefix_and_suffix(remove_boxes_keep_content(remove_text_box_only(gt_num_lst[0])))
        parse_result = number_check(model_num, gt_num)  #todo: check if this is correct

        # As an additional method of judgment, even if it returns False we can't say that the answer is wrong, it could be caused by an unreasonable extraction of numbers
        if parse_result is True:
            return True

    # Here we do the same thing to the whole string
    model_wo_text = remove_prefix_and_suffix(model_answer)
    gt_wo_text = remove_prefix_and_suffix(gt_answer)
    parse_result = number_check(model_wo_text, gt_wo_text)
    if parse_result is True:
        return True

    """
    #! Final try: using `sympy`
    res = check_tex_equiv(model_answer, gt_answer)
    if res is True:
        return True
    
    res = check_funcs_equiv(model_answer, gt_answer)
    if res is True:
        return True
    """

    # If none of the above ways can determine whether the answer is correct or incorrect, then return incorrect
    return False


def latex_equiv(model_answer: str, gt_answer: str) -> bool:
    """
    Check if two strings are equivalent in LaTeX format.
    """
    candidate_answers = [model_answer]
    
    conjunctions = ["\\text{ or }", "\\text{ and }"]
    for c in conjunctions:
        if c in model_answer:
            split = model_answer.split(c)
            for items in permutations(split):
                ans = " , ".join(items)
                if ans not in candidate_answers:
                    candidate_answers.append(ans)
            break
    
    for ans in candidate_answers:
        if _latex_equiv(ans, gt_answer):
            return True
    
    return False
