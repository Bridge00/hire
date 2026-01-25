
from utils.llm import get_llm_response
from utils.prompts import CODEGEN_SYS

class CodeGenerator:
    def __init__(self, model: str = "gpt-4o-mini", dataset: str = None):
        self.model = model
        self.dataset = dataset

    def generate(self, prompt: str, task_id: str = None, force: bool = False) -> str:
        """
        Generates code based on the given prompt.
        
        Args:
            prompt: The input problem description.
            task_id: The task identifier (for structured caching).
            
        Returns:
            The generated code as a string.
        """
        programming_language = "python"
        if self.dataset:
            ds_lower = self.dataset.lower()
            if "js" in ds_lower:
                programming_language = "javascript"
            elif "java" in ds_lower:
                programming_language = "java"
            elif "cpp" in ds_lower:
                programming_language = "c++"
            elif "go" in ds_lower:
                programming_language = "go"
        
        formatted_sys_prompt = CODEGEN_SYS.format(
            PROGRAM_LANGUAGE=programming_language.upper(),
            PROGRAM_LANGUAGE_LOWER=programming_language
        )
        
        return get_llm_response(sys_prompt=formatted_sys_prompt, 
                                user_prompt=prompt, 
                                model=self.model,
                                dataset=self.dataset,
                                task_id=task_id,
                                force=force)
