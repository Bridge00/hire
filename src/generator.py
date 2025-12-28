from abc import ABC, abstractmethod

class CodeGenerator(ABC):
    """
    Abstract base class for the Code Generator (G).
    Responsible for generating candidate code based on a prompt.
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generates code based on the given prompt.
        
        Args:
            prompt: The input problem description.
            
        Returns:
            The generated code as a string.
        """
        pass
