import platformdirs
from .base import Dataset
import os
import json
from dotenv import load_dotenv
load_dotenv()

def extract_asserts(text: str):
    lines = text.splitlines()
    return [line.strip() for line in lines if line.strip().startswith("assert")]

def split_balanced(text: str, delimiter: str, open_chars='([{', close_chars=')]}'):
    """Split text by delimiter but only if outside of balanced brackets and not in strings."""
    parts = []
    current = []
    depth = 0
    i = 0
    quote_char = None
    escaped = False
    
    while i < len(text):
        char = text[i]
        
        if escaped:
            current.append(char)
            escaped = False
            i += 1
            continue
            
        if char == '\\':
            escaped = True
            current.append(char)
            i += 1
            continue
            
        if quote_char:
            if char == quote_char:
                quote_char = None
            current.append(char)
            i += 1
            continue
        
        if char in "'\"`":
            quote_char = char
            current.append(char)
            i += 1
            continue
            
        if char in open_chars:
            depth += 1
        elif char in close_chars:
            depth -= 1
        
        if depth == 0 and text[i:i+len(delimiter)] == delimiter:
            parts.append("".join(current).strip())
            current = []
            i += len(delimiter)
            continue
        
        current.append(char)
        i += 1
    
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]

def find_balanced_block(text: str, pattern: str, open_char='{', close_char='}'):
    """Find the preamble and content of a balanced block starting after pattern, while respecting strings."""
    import re
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None, None
    
    # Find the first opening character associated with this match
    open_pos = text.find(open_char, match.start())
    if open_pos == -1:
        return None, None
    
    depth = 0
    quote_char = None
    escaped = False
    
    for i in range(open_pos, len(text)):
        char = text[i]
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if quote_char:
            if char == quote_char:
                quote_char = None
            continue
        if char in "'\"`":
            quote_char = char
            continue
            
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                # Content is between open_pos + 1 and i
                # Preamble is everything before the match.start()
                return text[:match.start()].strip(), text[open_pos+1:i].strip()
    return None, None

def extract_js_tests(text: str):
    import re
    
    # 1. Identify the block to analyze
    content = text
    preamble = ""
    match = re.search(r'=>\s*\{', text)
    if match:
        preamble, inner = find_balanced_block(text, r'=>\s*\{')
        if inner:
            content = inner
            
    # 2. Identify all console.assert calls and their depths
    calls = []
    i = 0
    while i < len(content):
        match = re.search(r'console\.assert', content[i:])
        if not match: break
        start_idx = i + match.start()
        
        # Calculate depth at start_idx
        depth = 0
        quote_char = None
        escaped = False
        for k in range(start_idx):
            char = content[k]
            if escaped: escaped = False; continue
            if char == '\\': escaped = True; continue
            if quote_char:
                if char == quote_char: quote_char = None
                continue
            if char in "'\"`": quote_char = char; continue
            if char in '({[': depth += 1
            elif char in ')}]': depth -= 1
            
        # Find end of this assert
        open_pos = content.find('(', start_idx)
        if open_pos != -1:
            inner_depth = 0
            quote_char = None
            escaped = False
            for j in range(open_pos, len(content)):
                c2 = content[j]
                if escaped:
                    escaped = False
                    continue
                if c2 == '\\':
                    escaped = True
                    continue
                if quote_char:
                    if c2 == quote_char:
                        quote_char = None
                    continue
                if c2 in "'\"`":
                    quote_char = c2
                    continue
                    
                if c2 == '(': inner_depth += 1
                elif c2 == ')':
                    inner_depth -= 1
                    if inner_depth == 0:
                        # Check for dangling comparison (e.g. console.assert(...) === ...)
                        # This handles malformed tests in some datasets
                        curr_end = j + 1
                        rest = content[curr_end:]
                        import re
                        dangling_match = re.match(r'\s*(===|==|!=|!==)', rest)
                        if dangling_match:
                            # If we see a dangling comparison, we consume until the next console.assert or EOF
                            # But we must stop before the next console.assert to avoid eating it.
                            # We can just look for the next console.assert in 'rest'
                            next_assert = re.search(r'console\.assert', rest)
                            if next_assert:
                                extra_end = curr_end + next_assert.start()
                            else:
                                extra_end = len(content)
                            curr_end = extra_end

                        calls.append({"text": content[start_idx:curr_end], "start": start_idx, "end": curr_end, "depth": depth})
                        i = curr_end
                        break
            else: i = start_idx + 1
        else: i = start_idx + 1

    if not calls:
        return [text.strip()]

    # 3. Decision: Split or Monolithic?
    # If ANY console.assert is nested (depth > 0), we treat the whole thing as one test to be safe.
    # Exception: if it's wrapped in () => { ... } we already unwrapped it, so depth 0 refers to the function body.
    if any(c["depth"] > 0 for c in calls):
        return [text.strip()]
        
    # 4. Splitting logic (for top-level asserts)
    # We want to keep everything that isn't a console.assert as "shared setup"
    results = []
    setup_code = ""
    last_end = 0
    for call in calls:
        # Add everything between last call and this call to setup
        segment = content[last_end:call["start"]].strip()
        if segment:
            setup_code += segment + ("\n" if not segment.endswith(";") else ";\n")
        
        # This test is everything we've gathered as setup + this call
        # We also prefix with the original preamble if we unwrapped (to maintain context like "testSortNumbers = () => {")
        # BUT wait, the preamble might be an incomplete statement. 
        # For JS assertions, we can usually just run the body if we've gathered all setup.
        
        full_test = f"{setup_code}\n{call['text']}"
        results.append(full_test.strip())
        last_end = call["end"]
        
    return results

def extract_cpp_tests(text: str):
    preamble, body = find_balanced_block(text, r'int main\s*\(\)')
    if not body:
        return [text.strip()]
    
    statements = split_balanced(body, ";")
    setup = []
    asserts = []
    for stmt in statements:
        if stmt.strip().startswith("assert"):
            asserts.append(stmt.strip() + ";")
        elif stmt.strip():
            setup.append(stmt.strip() + ";")
    
    setup_code = "\n".join(setup)
    results = []
    for a in asserts:
        # Construct a full main function to avoid evaluator wrapping issues with macros
        full_test = f"{preamble}\nint main() {{\n{setup_code}\n{a}\nreturn 0;\n}}"
        results.append(full_test.strip())
        
    if not results:
        # If no top-level asserts found (e.g. they are inside loops), 
        # return the monolithic original test
        return [text.strip()]
        
    return results

def extract_java_tests(text: str):
    # Java evaluator wraps it if "class Main" is missing.
    # To avoid multi-line string literal issues in the evaluator's AssertionError, 
    # we can try to return a full class Main, OR just rely on the evaluator and fix the evaluator.
    
    # 1. SPECIAL HANDLING: List<Boolean> pattern (Java 95, 111, 162)
    # We want granular tests, but we need to preserve setup code (variables defined before Arrays.asList).
    if "List<Boolean>" in text and "Arrays.asList" in text:
        preamble, body = find_balanced_block(text, r'Arrays\.asList', '(', ')')
        if body:
            # Preamble contains imports + class Main + void main + setup code.
            # We need to strip the class/method wrappers to extract just the setup code.
            import re
            
            # Remove package declaration if present (usually not in HumanEvalJava but good to be safe)
            setup_code = re.sub(r'package\s+[\w.]+;', '', preamble)
            
            # Remove imports from preamble (we will re-add standard imports)
            setup_code = re.sub(r'import\s+[\w.]+;', '', setup_code)
            
            # Remove class Main wrapper (tolerant regex)
            setup_code = re.sub(r'public\s+class\s+Main\s*\{', '', setup_code)
            
            # Remove main method wrapper
            setup_code = re.sub(r'public\s+static\s+void\s+main\s*\(String\[\]\s*args\)\s*(?:throws\s+[\w,\s]+)?\s*\{', '', setup_code)
            
            # Remove List<Boolean> correct = assignment part
            setup_code = re.sub(r'List\s*<\s*Boolean\s*>\s*correct\s*=\s*$', '', setup_code.strip())
            
            
            individual_tests = split_balanced(body, ",")
            results = []
            for t in individual_tests:
                # Reconstruct full Main class with imports, setup code, and single assertion
                # NOTE: setup_code likely already contains 'Solution s = new Solution();' so don't duplicate it.
                # However, if it doesn't, we might need it. But based on inspection, it does.
                
                # Fix escaping for the error message:
                # We want the Java string literal for the error message to contain the test case code.
                # If t contains quotes like 's.check("foo")', we want the java string to be "Test failed: s.check(\"foo\")"
                # So we replace " with \" to escape it for the Java string literal.
                assertion_msg_escaped = t.replace('"', '\\"')
                
                full_test = (
                    "import java.util.*;\nimport java.lang.*;\nimport java.security.*;\n"
                    "public class Main {\n"
                    "    public static void main(String[] args) throws Exception {\n"
                    f"        {setup_code}\n"
                    f"        if (!({t})) throw new AssertionError(\"Test failed: {assertion_msg_escaped}\");\n"
                    "    }\n"
                    "}"
                )
                results.append(full_test.strip())
            return results

    # 2. General extraction
    preamble, body = find_balanced_block(text, r'Arrays\.asList', '(', ')')
    if not body:
        return [text.strip()]
    
    individual_tests = split_balanced(body, ",")
    results = []
    for t in individual_tests:
        escaped_t = t.replace('"', '\\"')
        # Add throws Exception to handle checked exceptions (Java 162)
        full_test = f"public class Main {{ public static void main(String[] args) throws Exception {{ Solution s = new Solution(); if (!({t})) throw new AssertionError(\"Test failed\"); }} }}"
        results.append(full_test.strip())
    return results

def extract_go_tests(text: str):
    preamble, body = find_balanced_block(text, r'func Test\w+\(t \*testing\.T\)')
    if not body:
        return [text.strip()]
    
    # 2. Advanced Splitting: Statement-aware
    # Go tests structure:
    # - Setup (vars, helper funcs)
    # - Tests (assertions, loops with assertions)
    # We collect "Statements" (balanced code blocks).
    # If a statement contains an assertion, it's a test case.
    # Otherwise, it's global setup for subsequent tests.
    
    lines = body.splitlines()
    setup_lines = []
    results = []
    
    current_stmt = []
    brace_depth = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped: 
            if brace_depth > 0:
                current_stmt.append(line)
            continue

        current_stmt.append(line)
        
        # Check for braces (ignoring comments/strings lightly - typically OK for these benchmarks)
        open_braces = stripped.count('{')
        close_braces = stripped.count('}')
        
        brace_depth += (open_braces - close_braces)
        
        if brace_depth == 0:
            # End of a statement/block
            stmt_text = "\n".join(current_stmt)
            
            # Decide: Setup or Test?
            # Check for assert usage (ignoring setup calls)
            # assert.New(t) is setup. assert.Equal is test.
            has_assert = "assert." in stmt_text
            is_setup_assert = "assert.New" in stmt_text or (":=" in stmt_text and "assert" in stmt_text and not "assert." in stmt_text.split(":=")[-1]) # simplified check
            
            # Refined check:
            # If it contains "assert." but ONLY "assert.New", it's setup.
            # If it contains actual assertions (Equal, True, etc), it's a test.
            
            # Simple heuristic: "assert" variable usage usually implies test, EXCEPT declaration.
            # But the code uses `assert.Equal`.
            
            is_test = False
            if has_assert:
                # remove "assert.New" from consideration
                check_text = stmt_text.replace("assert.New", "")
                if "assert." in check_text:
                    is_test = True
            
            if is_test:
                # It's a granular test case
                # Accumulate all setup seen so far
                setup_code = "\n".join(setup_lines)
                idx = len(results)
                # Inject _ = assert safety
                full_test = f"{preamble}\nfunc TestGranular{idx}(t *testing.T) {{\n{setup_code}\n_ = assert\n{stmt_text}\n}}"
                results.append(full_test.strip())
            else:
                # It's setup (vars, helpers, or assert.New)
                setup_lines.append(stmt_text)
                
            current_stmt = [] # Reset for next statement

    if not results:
         return [text.strip()]
         
    return results
             
    if not results:
         # Fallback
         return [text.strip()]
         
    return results

class CodeData(Dataset):
    def __init__(self, task_name: str = None):
        # if root is None:

        #self.root = root
        if task_name.lower() == "leetcode": #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = 'leetcode-hard.jsonl'
        elif task_name.lower() == "humaneval": #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = 'HumanEval.jsonl'
        elif "evoeval" in task_name.lower(): #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = f'EvoEval_{task_name[task_name.find("_")+1:]}.jsonl'#'leetcode-hard.json'
        elif "core_eval" in task_name.lower(): #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = f'core_eval.jsonl'#'leetcode-hard.json'
        elif task_name.lower().startswith("apps_"):
            self.data_path = task_name.lower() + '.jsonl'
        else:
            self.data_path = task_name.lower() + '.jsonl'
        self.data_path = os.path.join(os.getenv("DATA_DIR"), self.data_path)
        print('loading', self.data_path)
        self._check_or_download_dataset()

        self.dataset = [json.loads(line) for line in open(self.data_path)]
        
        self._task_description = f'You will solve a hard coding problem from {task_name.lower()}. You will be given a prompt describing a problem. You need to write a function that passes all the tests.'

    def get_task_description(self):
        return self._task_description

    def _check_or_download_dataset(self):
        #data_path = #f"{self.root}/leetcode-hard.jsonl"
        #print(self.data_path, self.root)
        if os.path.exists(self.data_path):
            return
        
        # os.makedirs(f"{self.root}/", exist_ok=True)
        # import requests
        # url = "https://raw.githubusercontent.com/vinid/data/master/leetcode_with_tests.jsonl"
        # r = requests.get(url)
        # with open(data_path, 'wb') as f:
        #     f.write(r.content)

    def __getitem__(self, index):
        #print('index', index)
        #print('len of dataset', len(self.dataset))
        #print('type of dataset', type(self.dataset))
        #print('type of dataset[index]', type(self.dataset[index]))
        row = self.dataset[index]

        tests = row['test']
        if "apps_" in self.data_path.lower():
            # For APPS, 'test' is the io_data dictionary. We pass it as is.
            tests = row['test']
        elif self.data_path.lower().endswith("humaneval.jsonl") or self.data_path.lower().endswith("humaneval_py.jsonl") or "leetcode" in self.data_path.lower():
            tests = extract_asserts(row['test'])
        elif "humaneval_js" in self.data_path.lower():
            tests = extract_js_tests(row['test'])
        elif "humaneval_cpp" in self.data_path.lower():
            tests = extract_cpp_tests(row['test'])
        elif "humaneval_java" in self.data_path.lower():
            tests = extract_java_tests(row['test'])
        elif "humaneval_go" in self.data_path.lower():
            tests = extract_go_tests(row['test'])
        elif "humaneval_" in self.data_path.lower():
            # For other variants
            tests = [row['test']]
        else:
            tests = row['test']
            if not isinstance(tests, list):
                tests = [tests]
        # if "evoeval" in self.data_path.lower():
        #     tests = row["inputs"]
        # else:
        task_id = row["task_id"]
        if "/" in task_id:
            task_id = '_'.join(task_id.split('/'))
            
        return task_id, row["prompt"], tests, row['canonical_solution']

    def __len__(self):
        return len(self.dataset)

