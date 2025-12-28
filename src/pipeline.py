from typing import List, Dict, Any
from src.generator import CodeGenerator
from src.evaluator import Evaluator
import utils.llm as ul
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

    def run_experiment(self, dataset) -> List[Dict[str, Any]]:
        """
        Runs the experiment on the dataset.
        
        Args:
            dataset: CodeData instances
            
        Returns:
            List of result dictionaries.
        """
        results = []
        
        # Iterate over the dataset
        # CodeData __getitem__ returns: task_id, prompt, tests, canonical_solution
        for i, item in enumerate(dataset):
            
            # Unpack item based on expected format
            if isinstance(item, tuple) and len(item) >= 3:
                task_id, prompt, tests = item[0], item[1], item[2]
                canonical_solution = item[3] if len(item) > 3 else None
            else:
                print(f"Skipping item {i}: unknown format {type(item)}")
                continue

            if not prompt:
                continue
                
            print(f"Processing task {i} out of {len(dataset)} : Task ID: {task_id}...")
            
            # 1. Generate
            code = self.generator.generate(prompt)
            
            if self.evaluator is None:
                continue

            code = ul.clean_code(code)  
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
                "task_id": task_id,
                "prompt": prompt,
                "generated_code": code,
                "canonical_solution": canonical_solution,
                "LLM_evaluation": llm_evaluation,
                "execution_metrics": execution_metrics
            }
            results.append(result)
            
        return results
