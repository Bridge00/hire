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
            if not prompt:
                continue
                
            print(f"Processing item {i}...")
            
            # 1. Generate
            code = self.generator.generate(prompt)
            
            # 2. Evaluate
            metrics = self.evaluator.evaluate(prompt, code)
            
            # 3. Store
            result = {
                "item_index": i,
                "prompt": prompt,
                "generated_code": code,
                "metrics": metrics
            }
            results.append(result)
            
        return results
