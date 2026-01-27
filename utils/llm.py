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

import io
import tokenize

def strip_comments_preserve_docstrings(code: str, lang: str = 'py') -> str:
    """
    Remove all single-line comments from code, preserving docstrings (block comments) and strings.
    
    Args:
        code: The source code string.
        lang: 'py', 'cpp', 'java', 'js', 'go'.
    """
    if lang == 'py':
        out = []
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            for tok in tokens:
                tok_type, tok_str, start, end, line = tok
                # Skip actual comment tokens (single line in Python starts with #)
                if tok_type == tokenize.COMMENT:
                    continue
                out.append(tok)
            return tokenize.untokenize(out)
        except tokenize.TokenError:
            # Fallback if tokenization fails (e.g. partial code)
            return code
    
    elif lang in ['cpp', 'java', 'js', 'go']:
        # Regex to match:
        # 1. Strings (Double, Single, Backtick) -> Keep
        # 2. Block Comments (/* ... */) -> Keep (treated as docstrings)
        # 3. Single Line Comments (// ...) -> Remove
        
        # Note: Go uses backticks for raw strings. JS uses backticks for templates.
        # strings: "...", '...', `...`
        # We need to be careful with escaping in strings.
        
        pattern = r'("(?:\\.|[^\\"])*"|\'(?:\\.|[^\\\'])*\'|`[^`]*`|/\*[\s\S]*?\*/)|(//.*)'
        
        def replacer(match):
            # If group 1 matches (string or block comment), keep it.
            if match.group(1):
                return match.group(1)
            # If group 2 matches (single line comment), remove it (return empty).
            return ""
            
        return re.sub(pattern, replacer, code)
        
    return code