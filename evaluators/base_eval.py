from .docker_utils import DockerRunner

class BaseDockerEvaluator:
    def __init__(self, image_name="hire-evaluator"):
        self.runner = DockerRunner(image_name=image_name)

    def generate_feedback(self, success_tests, failed_tests):
        feedback = "**Tests that the code passed:**\n"
        if len(success_tests) == 0:
            feedback += "\nNo tests passed.\n"
        else:
            for test in success_tests:
                feedback += f"\n{test}"
        
        feedback += "\n\n**Tests that the code failed:**\n"
        if len(failed_tests) == 0:
            feedback += "\nNo tests failed.\n"
        else:
            for test in failed_tests:
                feedback += f"\n{test}"
        
        return feedback

    def evaluate(self, code, tests, dataset_name, supervised='supervised'):
        raise NotImplementedError("Subclasses must implement evaluate()")
