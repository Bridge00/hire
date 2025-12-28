
from utils.llm import get_llm_response
from utils.prompts import CODEGEN_SYS

class CodeGenerator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generates code based on the given prompt.
        
        Args:
            prompt: The input problem description.
            
        Returns:
            The generated code as a string.
        """
        
        return get_llm_response(sys_prompt=CODEGEN_SYS, 
                                user_prompt=prompt, 
                                model=self.model)
