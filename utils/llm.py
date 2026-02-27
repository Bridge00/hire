import os
import hashlib
import json
import re

from dotenv import load_dotenv

load_dotenv()

TOGETHER_AI_MODELS = [
    "Qwen/Qwen3-8B-Base",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
]


os.makedirs(os.getenv("CACHE_DIR"), exist_ok=True)

def _get_cache_key(sys_prompt: str, user_prompt: str, model: str) -> str:
    """Legacy hash-based cache key for backward compatibility."""
    unique_str = f"{model}:{sys_prompt}:{user_prompt}"
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def _sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename."""
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def _get_structured_cache_path(model: str, dataset: str = None, task_id: str = None, 
                                 eval_model: str = None, eval_prompt_type: str = None) -> str:
    """
    Get a human-readable cache path based on context.
    
    For code generation: dataset/code_gen_model/task_id.json
    For evaluation: dataset/code_gen_model/eval_model/eval_prompt_type/task_id.json
    """
    cache_dir = os.getenv("CACHE_DIR", ".cache")
    
    if not dataset or not task_id:
        # Fallback to hash-based if context is missing
        return None
    
    # Sanitize components
    dataset = _sanitize_filename(dataset)
    task_id = _sanitize_filename(task_id)
    model = _sanitize_filename(model)
    
    if eval_model and eval_prompt_type:
        # Evaluation cache: dataset/code_gen_model/eval_model/eval_prompt_type/task_id.json
        eval_model = _sanitize_filename(eval_model)
        eval_prompt_type = _sanitize_filename(eval_prompt_type)
        cache_path = os.path.join(cache_dir, dataset, model, eval_model, eval_prompt_type, f"{task_id}.json")
    else:
        # Code generation cache: dataset/code_gen_model/task_id.json
        cache_path = os.path.join(cache_dir, dataset, model, f"{task_id}.json")
    
    return cache_path

def get_cached_content(dataset: str, task_id: str, model: str, 
                       eval_model: str = None, eval_prompt_type: str = None) -> str:
    """
    Safely retrieve content from the structured cache.
    Returns None if not found, instead of making an API call.
    """
    cache_path = _get_structured_cache_path(
        model=model,
        dataset=dataset,
        task_id=task_id,
        eval_model=eval_model,
        eval_prompt_type=eval_prompt_type
    )
    
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f).get("content")
    return None

def get_llm_response(sys_prompt: str, user_prompt: str, model: str, 
                     dataset: str = None, task_id: str = None,
                     eval_model: str = None, eval_prompt_type: str = None,
                     force: bool = False) -> str:
    """
    Get LLM response with caching.
    
    Args:
        sys_prompt: System prompt
        user_prompt: User prompt
        model: Model name (for code generation) or eval model name
        dataset: Dataset name (optional, for structured cache)
        task_id: Task ID (optional, for structured cache)
        eval_model: Evaluation model (optional, for evaluation cache)
        eval_prompt_type: Type of evaluation prompt (e.g., 'vanilla', 'cj_analysis', 'cj_summary')
    
    Returns:
        LLM response content
    """
    
    # Try structured cache first
    structured_cache_path = _get_structured_cache_path(
        model=eval_model if eval_model else model,
        dataset=dataset,
        task_id=task_id,
        eval_model=eval_model,
        eval_prompt_type=eval_prompt_type
    )
    
    if not force and structured_cache_path and os.path.exists(structured_cache_path):
        print(f'Reading from structured cache: {structured_cache_path}')
        with open(structured_cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["content"]
    
    # Fallback to hash-based cache for backward compatibility
    cache_key = _get_cache_key(sys_prompt, user_prompt, model)
    hash_cache_path = os.path.join(os.getenv("CACHE_DIR"), f"{cache_key}.json")
    
    if not force and os.path.exists(hash_cache_path):
        print(f'Reading from hash cache: {hash_cache_path}')
        with open(hash_cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["content"]

    # Make API call
    if model in TOGETHER_AI_MODELS:
        from together import Together
        client = Together() 
    elif model in OPENAI_MODELS:
        from openai import OpenAI
        client = OpenAI()
    else:
        raise ValueError(f"Unknown model: {model}")
        
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
             "content": sys_prompt
        },
        {
            "role": "user",
            "content":  user_prompt
        }
    ]
    )
    content = response.choices[0].message.content
    
    # Save to structured cache if context is available
    if structured_cache_path:
        os.makedirs(os.path.dirname(structured_cache_path), exist_ok=True)
        with open(structured_cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f)
        print(f'Saved to structured cache: {structured_cache_path}')
    else:
        # Fallback to hash-based cache
        with open(hash_cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": content}, f)
        print(f'Saved to hash cache: {hash_cache_path}')
        
    return content

def clean_code(code):
    if isinstance(code, list):
        code = code[0]
    
    # Try to find code between markdown blocks
    # This regex looks for ```[language]\n[code]\n```
    match = re.search(r"```(?:\w+)?\n?(.*?)\n?```", code, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: if no code block but backticks exist, try to strip them
    if "```" in code:
        return code.split("```")[-2].strip()
        
    return code.strip()



import re
import io
import tokenize

def strip_js_line_comments_preserve_block_and_strings(code: str) -> str:
    i = 0
    n = len(code)
    out = []

    in_line_comment = False
    in_block_comment = False

    in_string = False
    string_quote = None  # ', ", `
    escape = False

    in_regex = False
    in_regex_charclass = False  # inside [...] within a regex

    def prev_non_ws(idx: int):
        j = idx - 1
        while j >= 0 and code[j].isspace():
            j -= 1
        return code[j] if j >= 0 else None

    # Heuristic: a `/` can start a regex after these contexts (or BOF).
    REGEX_PREFIX_CHARS = {
        None, '(', '=', '[', '{', ',', ':', ';',
        '!', '?', '+', '-', '*', '%', '&', '|', '^', '~',
        '\n'  # helps after line breaks
    }

    while i < n:
        ch = code[i]
        nxt = code[i + 1] if i + 1 < n else ''

        # Strip // ... until newline
        if in_line_comment:
            if ch == '\n':
                in_line_comment = False
                out.append(ch)
            i += 1
            continue

        # Keep /* ... */ blocks verbatim
        if in_block_comment:
            out.append(ch)
            if ch == '*' and nxt == '/':
                out.append(nxt)
                i += 2
                in_block_comment = False
            else:
                i += 1
            continue

        # Keep strings / template literals verbatim
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == string_quote:
                in_string = False
                string_quote = None
            i += 1
            continue

        # Keep regex literal verbatim, with correct handling of [...] char classes
        if in_regex:
            out.append(ch)
            if escape:
                escape = False
                i += 1
                continue
            if ch == '\\':
                escape = True
                i += 1
                continue

            if in_regex_charclass:
                if ch == ']':
                    in_regex_charclass = False
                i += 1
                continue

            # not in charclass
            if ch == '[':
                in_regex_charclass = True
                i += 1
                continue

            if ch == '/':
                in_regex = False  # end of regex literal; flags follow as normal chars
            i += 1
            continue

        # Start of string?
        if ch in ("'", '"', '`'):
            in_string = True
            string_quote = ch
            out.append(ch)
            i += 1
            continue

        # Start of comment?
        if ch == '/' and nxt == '/':
            in_line_comment = True
            i += 2
            continue

        if ch == '/' and nxt == '*':
            in_block_comment = True
            out.append(ch)
            out.append(nxt)
            i += 2
            continue

        # Start of regex literal?
        # Only if this '/' isn't starting a comment and prior context allows regex
        if ch == '/':
            prev = prev_non_ws(i)
            if prev in REGEX_PREFIX_CHARS and nxt not in ('/', '*'):
                in_regex = True
                in_regex_charclass = False
                out.append(ch)
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)



def strip_comments_preserve_docstrings(code: str, lang: str = 'py') -> str:
    """
    Remove all single-line comments from code, preserving docstrings (block comments) and strings.

    Args:
        code: The source code string.
        lang: 'py', 'python', 'cpp', 'c++', 'java', 'js', 'javascript', 'go', 'golang', etc.
    """
    lang = lang.lower()

    if lang in ['python', 'py']:
        out = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            for tok in tokens:
                tok_type, tok_str, start, end, line = tok
                if tok_type == tokenize.COMMENT:
                    continue
                out.append(tok)
            return tokenize.untokenize(out)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            return code

    # --- Go branch unchanged (your logic) ---
    if lang in ['go', 'golang']:
        lines = code.splitlines(keepends=True)
        out = []
        comment_block = []

        code_cleaner_pattern = r'("(?:\\.|[^\\"])*"|\'(?:\\.|[^\\\'])*\'|`[^`]*`)|(//.*)'

        def code_cleaner_replacer(match):
            if match.group(1): 
                return match.group(1)
            return ""

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//'):
                comment_block.append(line)
            else:
                is_decl = False
                if stripped:
                    first_word = stripped.split(' ')[0].split('(')[0]
                    if first_word in ['func', 'package', 'type', 'const', 'var']:
                        is_decl = True

                if is_decl:
                    out.extend(comment_block)
                comment_block = []

                clean_line = re.sub(code_cleaner_pattern, code_cleaner_replacer, line)
                out.append(clean_line)

        return "".join(out)

    # --- C/Java/C++/JS/TS/etc branch: use a real scanner ---
    if lang == 'js':
        return strip_js_line_comments_preserve_block_and_strings(code)
    if lang in ['cpp', 'java' ]:

        allow_backtick = lang in ['js', 'javascript', 'ts', 'typescript']

        i = 0
        n = len(code)
        out = []

        in_line_comment = False
        in_block_comment = False
        in_string = False
        string_quote = None  # one of ', ", `
        escape = False

        while i < n:
            ch = code[i]
            nxt = code[i + 1] if i + 1 < n else ''

            # If currently stripping a // comment: drop chars until newline
            if in_line_comment:
                if ch == '\n':
                    in_line_comment = False
                    out.append(ch)
                i += 1
                continue

            # If currently in /* ... */: KEEP everything until closing */
            if in_block_comment:
                out.append(ch)
                if ch == '*' and nxt == '/':
                    out.append(nxt)
                    i += 2
                    in_block_comment = False
                else:
                    i += 1
                continue

            # If currently inside a string literal: KEEP everything, respect escapes
            if in_string:
                out.append(ch)

                if escape:
                    escape = False
                    i += 1
                    continue

                # Backslash escapes apply to ' and " (and in JS also to backticks).
                # For simplicity, treat backslash as escape in all string modes.
                if ch == '\\':
                    escape = True
                    i += 1
                    continue

                if ch == string_quote:
                    in_string = False
                    string_quote = None

                i += 1
                continue

            # Not in comment/string: check start of string
            if ch in ("'", '"') or (allow_backtick and ch == '`'):
                in_string = True
                string_quote = ch
                out.append(ch)
                i += 1
                continue

            # Not in comment/string: check for // or /* 
            if ch == '/' and nxt == '/':
                in_line_comment = True
                i += 2
                continue

            if ch == '/' and nxt == '*':
                in_block_comment = True
                out.append(ch)
                out.append(nxt)
                i += 2
                continue

            # Normal char
            out.append(ch)
            i += 1

        return "".join(out)

    return code
