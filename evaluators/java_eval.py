from .base_eval import BaseDockerEvaluator

class JavaEvaluator(BaseDockerEvaluator):
    def evaluate(self, code, tests, dataset_name='Humaneval-Java', supervised='supervised'):
        success_tests = []
        failed_tests = []
        states = []

        # HumanEval Java has a Solution class and a Main class with tests.
        
        for test in tests:
            if "class Main" not in test:
                # If it's a raw boolean expression, wrap it in an assertion
                escaped_test = test.replace('"', '\\"')
                test_wrapped = f"public class Main {{ public static void main(String[] args) {{ Solution s = new Solution(); if (!({test})) throw new AssertionError(\"Test failed: {escaped_test}\"); }} }}"
            else:
                test_wrapped = test

            # Add necessary imports to Main.java as they are often missing in the benchmark test field
            # but used (e.g., Arrays.asList, List, ArrayList)
            java_imports = "import java.util.*;\nimport java.lang.*;\n\n"
            
            files = {
                "Solution.java": code,
                "Main.java": java_imports + test_wrapped
            }
            
            compile_cmd = "javac Solution.java Main.java"
            run_cmd = "java -ea Main"
            
            stdout, stderr, returncode = self.runner.run(files, f"{compile_cmd} && {run_cmd}")
            
            if returncode == 0:
                success_tests.append(test)
                states.append(True)
            else:
                error_msg = stderr.strip() or stdout.strip() or "Java compilation or runtime error"
                if supervised == 'supervised':
                    failed_tests.append(f"{test} # ERROR: {error_msg}")
                else:
                    failed_tests.append(f"{test} # ERROR: This unit test fails.")
                states.append(False)

        feedback = self.generate_feedback(success_tests, failed_tests)
        return tuple(states), feedback
