from typing import Dict, Any
from utils.llm import get_llm_response

class Evaluator:

    def __init__(self, model: str = "gpt-4o-mini", dataset: str = None, code_gen_model: str = None):
        self.model = model
        self.dataset = dataset
        self.code_gen_model = code_gen_model
   
    def evaluate(self, sys_prompt: str, user_prompt: str, task_id: str = None, eval_prompt_type: str = None) -> Dict[str, Any]:
        """
        Evaluates the generated code against the task prompt.
        
        Args:
            sys_prompt: System prompt for evaluation
            user_prompt: User prompt for evaluation
            task_id: The task identifier (for structured caching)
            eval_prompt_type: Type of evaluation prompt (e.g., 'vanilla', 'cj_analysis', 'cj_summary')
            
        Returns:
            A dictionary containing evaluation metrics (e.g., {"score": 1.0, "is_correct": True}).
        """
        return get_llm_response(sys_prompt=sys_prompt, 
                                user_prompt=user_prompt, 
                                model=self.model,
                                dataset=self.dataset,
                                task_id=task_id,
                                eval_model=self.model,
                                eval_prompt_type=eval_prompt_type)
