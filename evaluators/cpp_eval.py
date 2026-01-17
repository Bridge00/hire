from .base_eval import BaseDockerEvaluator

class CppEvaluator(BaseDockerEvaluator):
    def evaluate(self, code, tests, dataset_name='Humaneval-CPP', supervised='supervised'):
        success_tests = []
        failed_tests = []
        states = []

        # HumanEval CPP tests are usually full main functions.
        # If tests is a list, we run each.
        
        for test in tests:
            # Check if test already contains #include
            # Solution usually has includes from prompt or we might need to add common ones
            full_code = f"#include <iostream>\n#include <vector>\n#include <string>\n#include <algorithm>\n#include <cmath>\n#include <map>\n#include <set>\n#include <assert.h>\nusing namespace std;\n\n{code}\n\n{test}"
            
            # Write to solution.cpp, compile, and run
            files = {"solution.cpp": full_code}
            compile_cmd = "g++ -O3 solution.cpp -o solution"
            run_cmd = "./solution"
            
            # Run compilation
            stdout, stderr, returncode = self.runner.run(files, f"{compile_cmd} && {run_cmd}")
            
            if returncode == 0:
                success_tests.append(test)
                states.append(True)
            else:
                error_msg = stderr.strip() or stdout.strip() or "Compilation or Runtime error"
                if supervised == 'supervised':
                    failed_tests.append(f"{test} # ERROR: {error_msg}")
                else:
                    failed_tests.append(f"{test} # ERROR: This unit test fails.")
                states.append(False)

        feedback = self.generate_feedback(success_tests, failed_tests)
        return tuple(states), feedback
