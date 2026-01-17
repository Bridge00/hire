from .py_eval import PythonEvaluator
from .js_eval import JsEvaluator
from .cpp_eval import CppEvaluator
from .go_eval import GoEvaluator
from .java_eval import JavaEvaluator

def get_evaluator(dataset_name):
    dataset_name = dataset_name.lower()
    if "humaneval_js" in dataset_name:
        return JsEvaluator()
    elif "humaneval_cpp" in dataset_name:
        return CppEvaluator()
    elif "humaneval_go" in dataset_name:
        return GoEvaluator()
    elif "humaneval_java" in dataset_name:
        return JavaEvaluator()
    elif "leetcode" in dataset_name or "humaneval" in dataset_name or "py" in dataset_name:
        # Default to Python for leetcode and humaneval (which is usually python unless specified)
        return PythonEvaluator()
    else:
        # Fallback to Python but maybe we should warn
        return PythonEvaluator()
