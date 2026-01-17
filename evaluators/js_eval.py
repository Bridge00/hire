import json
from .base_eval import BaseDockerEvaluator

class JsEvaluator(BaseDockerEvaluator):
    def evaluate(self, code, tests, dataset_name='Humaneval-JS', supervised='supervised'):
        # For JS, we'll run each test case individually if possible, or all together.
        # HumanEval JS usually has a single 'test' string that calls multiple console.assert
        
        # Override console.assert to throw an error on failure
        js_setup = """
const console = require('console');
const originalAssert = console.assert;
console.assert = (condition, message) => {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
};
"""
        
        success_tests = []
        failed_tests = []
        states = []

        # If tests are provided as a list of independent assertions, we can run them one by one.
        # But if they are just one big string (as in HumanEval), we might need to parse them.
        # Let's assume 'tests' is a list of strings, where each string is a test.
        
        for test in tests:
            full_code = f"{js_setup}\n{code}\n{test}"
            stdout, stderr, returncode = self.runner.run({"solution.js": full_code}, "node solution.js")
            
            if returncode == 0:
                success_tests.append(test)
                states.append(True)
            else:
                error_msg = stderr.strip() or stdout.strip() or "Unknown error"
                if supervised == 'supervised':
                    failed_tests.append(f"{test} # ERROR: {error_msg}")
                else:
                    failed_tests.append(f"{test} # ERROR: This unit test fails.")
                states.append(False)

        feedback = self.generate_feedback(success_tests, failed_tests)
        return tuple(states), feedback
