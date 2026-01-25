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
                for m in re.findall(r'^import\s+((?:\w+\s+)?".*?")', text, re.MULTILINE):
                    import_lines.add(m.strip())

            get_imports(code)
            get_imports(test)
            
            # 3. Clean up source: remove package and import lines
            def clean_source(text):
                text = re.sub(r'^package\s+\w+\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^import\s*\((?:.|\n)*?\)\s*', '', text, flags=re.MULTILINE)
                text = re.sub(r'^import\s+(?:\w+\s+)?".*?"\s*', '', text, flags=re.MULTILINE)
                return text.strip()

            clean_code = clean_source(code)
            clean_test = clean_source(test)
            combined_clean_src = clean_code + "\n" + clean_test
            
            # --- Auto-Injection Logic ---
            # Common packages often missing in HumanEval Go
            # We use a stricter check: Pkg.Member (capitalized) to avoid matching "strings." at end of sentences in comments.
            # Example: "list of strings." should not trigger import "strings".
            # But "strings.Split" should.
            
            def uses_pkg(pkg, text):
                # Check for "pkg.Member" where Member starts with A-Z
                # This covers standard library usage which is always exported (Capitalized)
                return re.search(r'\b' + re.escape(pkg) + r'\.[A-Z]', text) is not None

            if uses_pkg("strings", combined_clean_src) and '"strings"' not in import_lines:
                import_lines.add('"strings"')
            if uses_pkg("math", combined_clean_src) and '"math"' not in import_lines:
                import_lines.add('"math"')
            if uses_pkg("sort", combined_clean_src) and '"sort"' not in import_lines:
                import_lines.add('"sort"')
            if uses_pkg("regexp", combined_clean_src) and '"regexp"' not in import_lines:
                import_lines.add('"regexp"')
            if uses_pkg("strconv", combined_clean_src) and '"strconv"' not in import_lines:
                import_lines.add('"strconv"')
            if uses_pkg("time", combined_clean_src) and '"time"' not in import_lines:
                import_lines.add('"time"')
            # math/rand is special, usage is 'rand.Intn' etc.
            if uses_pkg("rand", combined_clean_src) and '"math/rand"' not in import_lines:
                import_lines.add('"math/rand"')

            # Filter out problematic imports
            filtered_imports = []
            for imp in import_lines:
                if "github.com/stretchr/testify/assert" in imp: continue
                if '"testing"' in imp: continue
                filtered_imports.append(imp)
            
            # --- Unused Import Cleaning ---
            # Go is strict about unused imports. We need to be careful.
            # A simple heuristic: check if the package name appears in the source code.
            # This is not perfect but covers 99% of cases.
            final_imports = []
            for imp in filtered_imports:
                pkg_name = imp.strip('"') # "math" -> math
                if "/" in pkg_name: 
                    # "math/rand" -> rand
                    pkg_name = pkg_name.split("/")[-1]
                
                # Special cases where package usage doesn't match import name exactly
                # But for standard libs above, it usually does.
                # Also, we ALWAYS keep fmt, reflect, os because our harness uses them.
                if pkg_name in ["fmt", "reflect", "os"]:
                    final_imports.append(imp)
                    continue

                # Check usage in CLEANED source using strict check
                # Note: If the code ALREADY had the import, we should be slightly more lenient
                # because the user might have aliased it or used it weirdly?
                # But for safely removing "strings" when it's only in a comment, strict is better.
                # If legitimate usage involves non-capitalized member (impossible for external pkgs), we might break it.
                # But standard libs are fine.
                
                if uses_pkg(pkg_name, combined_clean_src) or \
                   re.search(r'\b' + re.escape(pkg_name) + r'\b', combined_clean_src): 
                   # Wait, the second clause `pkg_name + \b` would match "strings" in "list of strings."
                   # We MUST remove the broad check if we want to fix Go_158.
                   # BUT, we need to allow `var _ = strings.Split`.
                   # `uses_pkg` checks `strings.S`.
                   # What if they use `type Alias = strings.Builder`? `strings.B` matched.
                   # What if they use `strings` in a way that doesn't have a dot?
                   # No, to access it you need a dot. or `.` import (which we don't do).
                   # So `pkg.Member` is required.
                   
                   # However, I should be careful.
                   # Let's trust `uses_pkg` for the standard libs we know.
                   if pkg_name in ["strings", "math", "sort", "regexp", "strconv", "time", "rand"]:
                       if uses_pkg(pkg_name, combined_clean_src):
                           final_imports.append(imp)
                   else:
                       # Fallback for others (like internal ones if any?)
                       if re.search(r'\b' + re.escape(pkg_name) + r'\.', combined_clean_src):
                           final_imports.append(imp)

            import_block = "import (\n" + "\n".join(final_imports) + "\n)"

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
func (a *mockAssert) Len(object interface{}, length int, msg ...interface{}) {
    v := reflect.ValueOf(object)
    if v.Len() != length { os.Exit(1) }
}
func (a *mockAssert) ElementsMatch(listA, listB interface{}, msg ...interface{}) {
    if !reflect.DeepEqual(listA, listB) { os.Exit(1) }
}

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
