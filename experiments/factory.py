from typing import List, Dict, Any
from src.generator import CodeGenerator
from src.evaluator import Evaluator
from src.evaluator import HIREEvaluator



class VanillaEvaluator(Evaluator):
    """
    Represents a baseline evaluation method (e.g., 'CodeJudge' or 'Direct execution').
    """
    def evaluate(self, prompt: str, code: str) -> Dict[str, Any]:
        # Placeholder logic
        score = 0.8
        return {
            "score": score,
            "is_correct": score > 0.5,
            "method": "Vanilla"
        }

# --- Factory Functions ---

def get_dataset(dataset_name: str) -> List[Dict[str, str]]:
    if dataset_name == "dummy":
        return [{"prompt": "Write hello world"}, {"prompt": "Sort a list"}]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_generator() -> CodeGenerator:
    return DummyGenerator()

def get_evaluator(method_name: str, use_hire: bool) -> Evaluator:
    if use_hire:
        print(f"Initializing HIRE Evaluator (Base method: {method_name})")
        # In a real scenario, method_name might configure the Judge
        generator = get_generator() 
        summarizer = DummySummarizer()
        judge = DummyJudge()
        return HIREEvaluator(generator, summarizer, judge)
    else:
        if method_name == "vanilla":
            return VanillaEvaluator()
        else:
            raise ValueError(f"Unknown evaluation method: {method_name}")
