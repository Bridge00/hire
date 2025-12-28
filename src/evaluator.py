from abc import ABC, abstractmethod
from typing import Dict, Any

class Evaluator(ABC):
    """
    Abstract base class for an Evaluator.
    """
    
    @abstractmethod
    def evaluate(self, prompt: str, code: str) -> Dict[str, Any]:
        """
        Evaluates the generated code against the prompt.
        
        Args:
            prompt: The original problem description.
            code: The generated code to evaluate.
            
        Returns:
            A dictionary containing evaluation metrics (e.g., {"score": 1.0, "is_correct": True}).
        """
        pass
