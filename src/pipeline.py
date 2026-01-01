from typing import List, Dict, Any
from src.generator import CodeGenerator
from src.evaluator import Evaluator
import utils.llm as ul
class PipelineRunner:
    """
    Runs an experimental pipeline:
    For each item in dataset:
        1. Generate code (using G)
        2. LLM Evaluate code (using E)
        3. Collect results
    """
    
    def __init__(self, generator: CodeGenerator, evaluator: Evaluator, no_eval : bool, execute_solution: bool, dataset_name: str):
        self.generator = generator
        self.evaluator = evaluator
        self.no_eval = no_eval
        self.execute_solution = execute_solution
        self.dataset_name = dataset_name

    def run_experiment(self, dataset, py_evaluator) -> List[Dict[str, Any]]:
        """
        Runs the experiment on the dataset.
        
        Args:
            dataset: CodeData instances
            
        Returns:
            List of result dictionaries.
        """
        exec_results = []
        #llm_eval_results = []
        
        # Iterate over the dataset
        # CodeData __getitem__ returns: task_id, prompt, tests, canonical_solution
        for i, item in enumerate(dataset):
            #print(item.keys())
            
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
            
            if not self.execute_solution:
                # 1. Generate
                code = self.generator.generate(prompt, task_id=task_id)
                code = ul.clean_code(code)  
            else:
                code = canonical_solution
            
            
             
            # 3. Execution Metrics (with Cache)
            #execution_metrics = {}
            if tests:
                state, feedback = py_evaluator.evaluate(code, tests, self.dataset_name)

                exec_results.append({
                    "task_id": task_id,
                    "prompt": prompt,
                    "generated_code": code,
                    "canonical_solution": canonical_solution,
                    "state": state,
                    "feedback": feedback,
                    "pass_rate": sum(state) / len(state) if state else 0.0
                })

            if self.evaluator is not None and not self.no_eval:
                self.evaluator.evaluate(prompt, code)

        return exec_results

