import sys
import os

# Ensure we can import the hire package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator import CodeGenerator
from src.evaluator import Evaluator
from src.pipeline import PipelineRunner

# Mock Implementations
class MockGenerator(CodeGenerator):
    def generate(self, prompt: str) -> str:
        return f"print('Mock code for {prompt}')"

class MockEvaluator(Evaluator):
    def evaluate(self, prompt: str, code: str) -> dict:
        return {"score": 1.0, "is_correct": True}

def test_pipeline():
    print("Testing Pipeline...")
    
    dataset = [
        {"prompt": "Task 1"},
        {"prompt": "Task 2"}
    ]
    
    generator = MockGenerator()
    evaluator = MockEvaluator()
    runner = PipelineRunner(generator, evaluator)
    
    results = runner.run_experiment(dataset)
    
    print(f"Results: {results}")
    
    assert len(results) == 2
    assert results[0]["generated_code"] == "print('Mock code for Task 1')"
    assert results[0]["metrics"]["score"] == 1.0
    
    print("Pipeline test passed!")

if __name__ == "__main__":
    test_pipeline()
