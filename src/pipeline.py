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
    
    def __init__(self, generator: CodeGenerator, evaluator: Evaluator, no_eval : bool, execute_solution: bool, dataset_name: str, force: bool = False, eval_source: str = None):
        self.generator = generator
        self.evaluator = evaluator
        self.no_eval = no_eval
        self.execute_solution = execute_solution
        self.dataset_name = dataset_name
        self.force = force
        self.eval_source = eval_source

    def _process_task(self, i, item, py_evaluator, total_tasks):
        # Unpack item based on expected format
        if isinstance(item, tuple) and len(item) >= 4:
            task_id, prompt, tests, canonical_solution = item[0], item[1], item[2], item[3]
            full_item = item[4] if len(item) > 4 else {}
        else:
            print(f"Skipping item {i}: unknown format {type(item)}")
            return None

        if not prompt:
            return None
            
        print(f"Processing task {i} out of {total_tasks} : Task ID: {task_id}...")
        
        if self.eval_source:
             if self.eval_source not in full_item:
                 print(f"Warning: eval_source '{self.eval_source}' not found for task {task_id}. Skipping.")
                 return None
             code = prompt + full_item[self.eval_source]
        elif not self.execute_solution:
            # 1. Generate
            code = self.generator.generate(prompt, task_id=task_id, force=self.force)
            code = ul.clean_code(code)  
        else:
            code = prompt + canonical_solution
        
        # 3. Execution Metrics (with Cache)
        if tests:
            state, feedback = py_evaluator.evaluate(code, tests, self.dataset_name)

            result = {
                "task_id": task_id,
                "prompt": prompt,
                "generated_code": code,
                "canonical_solution": canonical_solution,
                "state": state,
                "feedback": feedback,
                "pass_rate": sum(state) / len(state) if state else 0.0
            }
            
            if self.evaluator is not None and not self.no_eval:
                self.evaluator.evaluate(prompt, code)
            
            return result
        return None

    def run_experiment(self, dataset, py_evaluator, parallel=False, num_workers=4) -> List[Dict[str, Any]]:
        """
        Runs the experiment on the dataset.
        
        Args:
            dataset: CodeData instances
            py_evaluator: Evaluator instance
            parallel: Whether to run in parallel
            num_workers: Number of workers for parallel execution
            
        Returns:
            List of result dictionaries.
        """
        exec_results = []
        
        total_tasks = len(dataset)
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm
            
            print(f"Running in parallel with {num_workers} workers...")
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self._process_task, i, item, py_evaluator, total_tasks) for i, item in enumerate(dataset)]
                for future in tqdm(futures, total=total_tasks):
                    res = future.result()
                    if res:
                        exec_results.append(res)
        else:
            for i, item in enumerate(dataset):
                res = self._process_task(i, item, py_evaluator, total_tasks)
                if res:
                    exec_results.append(res)

        return exec_results
