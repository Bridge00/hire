
from utils.llm import get_llm_response
from utils.prompts import CODEGEN_SYS

class CodeGenerator:
    def __init__(self, model: str = "gpt-4o-mini", dataset: str = None):
        self.model = model
        self.dataset = dataset

    def generate(self, prompt: str, task_id: str = None) -> str:
        """
        Generates code based on the given prompt.
        
        Args:
            prompt: The input problem description.
            task_id: The task identifier (for structured caching).
            
        Returns:
            The generated code as a string.
        """
        
        return get_llm_response(sys_prompt=CODEGEN_SYS, 
                                user_prompt=prompt, 
                                model=self.model,
                                dataset=self.dataset,
                                task_id=task_id)
