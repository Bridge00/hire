from together import Together
from openai import OpenAI

TOGETHER_AI_MODELS = [
    "Qwen/Qwen3-8B-Base",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
]

def get_llm_response(prompt: str, model : str ) -> str:
    
    if model in TOGETHER_AI_MODELS:
    
        client = Together() 
    elif model in OPENAI_MODELS:
        client = OpenAI()
    else:
        raise ValueError(f"Unknown model: {model}")
    response = client.chat.completions.create(
    model=model,
    messages=[
      {
        "role": "user",
        "content":  prompt
      }
    ]
    )
    return response.choices[0].message.content