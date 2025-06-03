import os, json, re
import random
from typing import List, Dict, Tuple
from collections import defaultdict

from .toolkit_for_MATH.latex_answer_check import latex_equiv as latex_equiv


class Evaluator:
    def __init__(self, num_last_chars_for_eval=128) -> None:
        self.num_last_chars_for_eval = num_last_chars_for_eval

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None
    
    def _last_boxed_only_string(self, string):
        if string is None:
            return None

        assert isinstance(string, str)

        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    def _remove_boxed(self, s):
        if s is None:
            return None

        assert isinstance(s, str)

        pattern = r'\\boxed\{(.*)\}'
        results = re.findall(pattern, s)

        if results:
            return results[-1]

        return None

    def validate_model_completion(self, completion: str) -> bool:
        return True
    
    def validate_model_answer(self, answer: str) -> bool:
        if answer is None:
            return False
        
        return True

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        if not self.validate_model_answer(answer_a) or not self.validate_model_answer(answer_b):
            return None
        
        return self._check_answers_equiv(answer_a, answer_b) or self._check_answers_equiv(answer_b, answer_a)
    
    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        """Extract the answer from the gold solution."""
        return self._extract_answer_from_gold_solution(solution)
    
    def _extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        """Extract the answer from the model completion."""
        if not self.validate_model_completion(completion):
            return None
        
        return self._extract_answer_from_model_completion(completion)
        
    def _extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError
    
    # def find_majority_completion_and_answer(self, list_of_completions: List[str]) -> str:
    #     """Find the majority completion from a list of completions."""
    #     if not list_of_completions:
    #         return None, None
        
    #     completion_counter = defaultdict(int)
    #     for completion in list_of_completions:
    #         answer = self.extract_answer_from_model_completion(completion)
            
    #         answer_already_exists = False
    #         for existing_completion in completion_counter.keys():
    #             existing_answer = self.extract_answer_from_model_completion(existing_completion)
    #             if self.check_answers_equiv(answer, existing_answer):
    #                 answer_already_exists = True
    #                 completion_counter[existing_completion] += 1
    #                 break
            
    #         if not answer_already_exists:
    #             completion_counter[completion] += 1
        
    #     majority_completion = max(completion_counter, key=completion_counter.get)
    #     return majority_completion, self.extract_answer_from_model_completion(majority_completion)
    
    def find_majority_completion_and_answer(self, candidates):
        """
        Finds the majority answer among the candidates using the evaluator's equivalence check.

        Args:
            candidates (List[str]): A list of candidate answer strings.

        Returns:
            str: The majority answer. If no clear majority is found, returns the first candidate.
        """
        if not candidates:
            return None
        
        assert isinstance(candidates, list)
        assert all(isinstance(x, str) for x in candidates)

        clusters = []  # Each cluster is a list of equivalent answers

        for candidate in candidates:
            candidate_answer = self.extract_answer_from_model_completion(
                candidate[-self.num_last_chars_for_eval :]
            )
            placed = False
            for cluster in clusters:
                representative_answer = self.extract_answer_from_model_completion(
                    cluster[0][-self.num_last_chars_for_eval :]
                )
                if self.check_answers_equiv(candidate_answer, representative_answer):
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])

        # Find the cluster with the maximum number of candidates
        majority_cluster = max(clusters, key=lambda cluster: len(cluster))
        majority_completion = majority_cluster[0]
        majority_answer = self.extract_answer_from_model_completion(
            majority_completion[-self.num_last_chars_for_eval :]
        )

        return majority_completion, majority_answer


class GSM8KEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None
        
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        return correct
    
    def validate_model_completion(self, completion: str) -> bool:
        if isinstance(completion, str) and "\\boxed" in completion:
            return True

        return False
    
    def _extract_answer_from_gold_solution(self, solution: str | float):
        if isinstance(solution, float):
            return str(solution)
        elif isinstance(solution, str):
            return solution.split("#### ")[-1].strip()
        else:
            raise ValueError("Invalid type of gold solution")
    
    def _extract_answer_from_model_completion(self, completion: str):
        ans = self._remove_boxed(completion)
        return ans if ans else None


class MATHEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None

        if answer_a == "" or answer_b == "":
            return False

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a.lower() == answer_b.lower():
            return True

        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            print(e)
            res = False

        return res

    def validate_model_completion(self, completion: str) -> bool:
        if isinstance(completion, str) and "\\boxed" in completion:
            return True
        
        return False

    def _extract_answer_from_gold_solution(self, solution: str):
        box = self._last_boxed_only_string(solution)
        ans = self._remove_boxed(box)
        return ans if ans else None

    def _extract_answer_from_model_completion(self, completion: str):
        box = self._last_boxed_only_string(completion)
        ans = self._remove_boxed(box)
        return ans if ans else None


class OlympiadEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def _check_answers_equiv(self, answer_a: str, answer_b: str):
        if answer_a is None or answer_b is None:
            return None

        if answer_a == "" or answer_b == "":
            return False

        answer_a = answer_a.strip()
        answer_b = answer_b.strip()

        if answer_a.lower() == answer_b.lower():
            return True

        try:
            res = latex_equiv(answer_a, answer_b)
        except Exception as e:
            print(e)
            res = False

        return res

    def _extract_answer_from_gold_solution(self, solution: str):
        answer = solution.strip("$")
        return answer

    def _extract_answer_from_model_completion(self, completion: str):
        box = self._last_boxed_only_string(completion)
        ans = self._remove_boxed(box)
        return ans if ans else None


EvaluatorMapping = {
    "GSM8K": GSM8KEvaluator,
    "GSM_Hard": GSM8KEvaluator,
    "MATH": MATHEvaluator,
    "MATH500": MATHEvaluator,
    "ScaleQuest": MATHEvaluator,
    "ScaleQuestLabeled": MATHEvaluator,
    "OpenMathInstruct": MATHEvaluator,
    "AMC2022": MATHEvaluator,
    "AMC2023": MATHEvaluator,
    "AMC": MATHEvaluator,
    "AIME": MATHEvaluator,
    "AIME2023": MATHEvaluator,
    "AIME2024": MATHEvaluator,
    "OlympiadBench": OlympiadEvaluator,
    "NuminaMath": MATHEvaluator,
}


def get_evaluator(task_name: str="MATH500") -> Evaluator:
    if task_name in EvaluatorMapping:
        return EvaluatorMapping[task_name]()
    else:
        if any(x in task_name for x in ["OpenMathInstruct", "MATH", "ScaleQuest", "NuminaMath"]):
            return MATHEvaluator()
        else:
            raise ValueError(f"Task name {task_name} not found in the evaluator mapping")
