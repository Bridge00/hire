from typing import Dict, Any
from utils.llm import get_llm_response

class Evaluator:

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
   
    def evaluate(self, sys_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Evaluates the generated code against the task prompt.
        
        Args:
            prompt: The original problem description.
            code: The generated code to evaluate.
            
        Returns:
            A dictionary containing evaluation metrics (e.g., {"score": 1.0, "is_correct": True}).
        """
        return get_llm_response(sys_prompt=sys_prompt, user_prompt=user_prompt, model=self.model)