import re
from .base_eval import BaseDockerEvaluator

class GoEvaluator(BaseDockerEvaluator):
    def evaluate(self, code, tests, dataset_name='Humaneval-Go', supervised='supervised'):
        success_tests = []
        failed_tests = []
        states = []

        # Mocking testify/assert for common methods:
        go_mock_assert = """
package main
import "fmt"
import "reflect"
import "os"

type mockT struct{}
func (t *mockT) Errorf(format string, args ...interface{}) {
    fmt.Fprintf(os.Stderr, format, args...)
    os.Exit(1)
}
func (t *mockT) FailNow() { os.Exit(1) }

type mockAssert struct { t *mockT }
func (a *mockAssert) Equal(expected, actual interface{}, msgAndArgs ...interface{}) {
    if !reflect.DeepEqual(expected, actual) {
        fmt.Fprintf(os.Stderr, "Expected %v, got %v\\n", expected, actual)
        a.t.FailNow()
    }
}
func (a *mockAssert) New(t *mockT) *mockAssert { return &mockAssert{t} }
var assert = &mockAssert{&mockT{}}
"""

        for test in tests:
            # 1. Extract function name to run
            test_match = re.search(r"func (Test\w+)\(t \*testing\.T\)", test)
            if test_match:
                test_func_name = test_match.group(1)
                run_test_code = f"func main() {{ {test_func_name}(&testingT{{}}) }}"
            else:
                # If it's not a Test function, we assume it's code that can go in main
                run_test_code = f"func main() {{\n    {test}\n}}"

            # 2. Extract and Deduplicate Imports
            # We must handle both single-line and multi-line imports
            import_lines = set()
            # Our required imports for the mock
            import_lines.add('"fmt"')
            import_lines.add('"reflect"')
            import_lines.add('"os"')
            
            def get_imports(text):
                # Multi-line: import (...)
                for m in re.findall(r'import\s*\((.*?)\)', text, re.DOTALL):
                    for line in m.split('\n'):
                        line = line.strip()
                        if line and not line.startswith("//"):
                            import_lines.add(line)
                # Single-line: import "foo" or import alias "foo"
                # Use a regex that doesn't match inside the ( ) blocks we already processed
                # This is a bit simplified but should work for HumanEval
                for m in re.findall(r'^import\s+((?:\w+\s+)?".*?")', text, re.MULTILINE):
                    import_lines.add(m.strip())

            get_imports(code)
            get_imports(test)

            # Filter out problematic imports
            filtered_imports = []
            for imp in import_lines:
                if "github.com/stretchr/testify/assert" in imp: continue
                if '"testing"' in imp: continue
                filtered_imports.append(imp)
            
            import_block = "import (\n" + "\n".join(filtered_imports) + "\n)"

            # 3. Clean up source: remove package and import lines
            def clean_source(text):
                text = re.sub(r'^package\s+\w+\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^import\s*\((?:.|\n)*?\)\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^import\s+(?:\w+\s+)?".*?"\s*', '', text, flags=re.MULTILINE)
                return text.strip()

            clean_code = clean_source(code)
            clean_test = clean_source(test)
            # Replace *testing.T with our mock
            clean_test = clean_test.replace("*testing.T", "*testingT")

            # 4. Assembly
            boilerplate_decls = """
type testingT struct{}
func (t *testingT) Errorf(format string, args ...interface{}) { fmt.Fprintf(os.Stderr, format, args...); os.Exit(1) }
func (t *testingT) FailNow() { os.Exit(1) }
func (t *testingT) Fatalf(format string, args ...interface{}) { fmt.Fprintf(os.Stderr, format, args...); os.Exit(1) }
func (t *testingT) Logf(format string, args ...interface{}) { }

type mockAssert struct {}
func (a *mockAssert) New(t interface{}) *mockAssert { return a }
func (a *mockAssert) Equal(expected, actual interface{}, msg ...interface{}) {
    if !reflect.DeepEqual(expected, actual) {
        fmt.Fprintf(os.Stderr, "Expected %v, got %v\\n", expected, actual)
        os.Exit(1)
    }
}
func (a *mockAssert) True(value bool, msg ...interface{}) { if !value { fmt.Fprintln(os.Stderr, "Expected true, got false"); os.Exit(1) } }
func (a *mockAssert) False(value bool, msg ...interface{}) { if value { fmt.Fprintln(os.Stderr, "Expected false, got true"); os.Exit(1) } }
func (a *mockAssert) NotNil(value interface{}, msg ...interface{}) { if value == nil { os.Exit(1) } }
func (a *mockAssert) Nil(value interface{}, msg ...interface{}) { if value != nil { os.Exit(1) } }

var assert = &mockAssert{}
"""
            # To avoid "imported and not used" for standard imports like math/sort which might be in the code
            # but we don't know for sure if they are used, we can't easily fix it without a two-pass.
            # But usually HumanEval solutions use all their imports.
            # If it fails, we will see.
            
            final_code = f"package main\n{import_block}\n{clean_code}\n{boilerplate_decls}\n{clean_test}\n{run_test_code}"
            
            stdout, stderr, returncode = self.runner.run({"solution.go": final_code}, "go run solution.go")
            
            if returncode == 0:
                success_tests.append(test)
                states.append(True)
            else:
                error_msg = stderr.strip() or stdout.strip() or "Go execution error"
                if supervised == 'supervised':
                    failed_tests.append(f"{test} # ERROR: {error_msg}")
                else:
                    failed_tests.append(f"{test} # ERROR: This unit test fails.")
                states.append(False)

        feedback = self.generate_feedback(success_tests, failed_tests)
        return tuple(states), feedback
