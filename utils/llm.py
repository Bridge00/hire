import os
import hashlib
import json

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
    unique_str = f"{model}:{sys_prompt}:{user_prompt}"
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def get_llm_response(sys_prompt: str, user_prompt: str, model : str ) -> str:
    
    # Check cache
    cache_key = _get_cache_key(sys_prompt, user_prompt, model)
    cache_path = os.path.join(os.getenv("CACHE_DIR"), f"{cache_key}.json")
    print(cache_path)
    if os.path.exists(cache_path):
        print('reading in cache data')
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["content"]

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
    
    # Save to cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"content": content}, f)
        
    return content

def clean_code(code):

    if isinstance(code, list):
        return code[0].split("```python")[1].split("```")[0]
    else:
        return code.split("```python")[1].split("```")[0] 