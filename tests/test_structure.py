import sys
import os

# Ensure we can import the hire package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hire.models import HierarchicalPlan, CodeStep
from src.generator import CodeGenerator
from hire.components.summarizer import Summarizer
from hire.components.judge import Judge
from hire.evaluator import HIREEvaluator

# Mock Implementations for Testing
class MockGenerator(CodeGenerator):
    def generate(self, prompt: str) -> str:
        return "def hello(): print('world')"

class MockSummarizer(Summarizer):
    def summarize(self, code: str) -> HierarchicalPlan:
        return HierarchicalPlan(steps=[
            CodeStep(code_segment="def hello():", explanation="Define function"),
            CodeStep(code_segment="print('world')", explanation="Print output")
        ])

class MockJudge(Judge):
    def evaluate_plan(self, prompt: str, plan: HierarchicalPlan) -> float:
        return 1.0

    def evaluate_implementation(self, prompt: str, step: CodeStep) -> float:
        return 1.0

def test_structure():
    print("Testing HIRE structure...")
    
    # Instantiate components
    generator = MockGenerator()
    summarizer = MockSummarizer()
    judge = MockJudge()
    
    # Instantiate Evaluator
    evaluator = HIREEvaluator(generator, summarizer, judge)
    
    # Run evaluation
    result = evaluator.run("Write a hello world function")
    
    print(f"Result: {result}")
    
    assert result.overall_correctness_score == 1.0
    assert len(result.partial_correctness_scores) == 2
    assert result.is_correct == True
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_structure()
