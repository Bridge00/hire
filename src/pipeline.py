from typing import List, Dict, Any
from src.generator import CodeGenerator
from src.evaluator import Evaluator

class PipelineRunner:
    """
    Runs an experimental pipeline:
    For each item in dataset:
        1. Generate code (using G)
        2. Evaluate code (using E)
        3. Collect results
    """
    
    def __init__(self, generator: CodeGenerator, evaluator: Evaluator):
        self.generator = generator
        self.evaluator = evaluator

    def run_experiment(self, dataset: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Runs the experiment on the dataset.
        
        Args:
            dataset: List of dicts, each must have a "prompt" key.
            
        Returns:
            List of result dictionaries.
        """
        results = []
        
        for i, item in enumerate(dataset):
            prompt = item.get("prompt")
            tests = item.get("tests", [])
            
            if not prompt:
                continue
                
            print(f"Processing item {i}...")
            
            # 1. Generate
            code = self.generator.generate(prompt)
            
            if self.evaluator is None:
                continue
            # 2. Evaluate (LLM)
            llm_evaluation = self.evaluator.evaluate(prompt, code)
            
            # 3. Execution Metrics (with Cache)
            execution_metrics = {}
            if tests:
                state, feedback = self.py_evaluator.evaluate(code, tests)
                execution_metrics = {
                        "state": state,
                        "feedback": feedback,
                        "pass_rate": sum(state) / len(state) if state else 0.0
                    }


            # 4. Store
            result = {
                "item_index": i,
                "prompt": prompt,
                "generated_code": code,
                "LLM_evaluation": llm_evaluation,
                "execution_metrics": execution_metrics
            }
            results.append(result)
            
        return results
