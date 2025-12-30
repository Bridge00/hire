import argparse
import json
import os
import sys
import re
from dotenv import load_dotenv

load_dotenv()


def process_and_cache_results(results_file: str, dataset: str, code_gen_model: str, 
                                eval_model: str = None, eval_prompt_type: str = None, 
                                cache_dir: str = ".cache"):
    """
    Process batch results and save to structured cache.
    Handles both code generation and evaluation results.
    
    Args:
        results_file: Path to the batch results .jsonl file
        dataset: Dataset name (e.g., 'leetcode')
        code_gen_model: Code generation model (e.g., 'gpt-4o-mini')
        eval_model: Evaluation model (optional, for evaluation results)
        eval_prompt_type: Type of evaluation (optional, for evaluation results)
        cache_dir: Base cache directory
    """
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    print(f"Dataset: {dataset}")
    print(f"Code Gen Model: {code_gen_model}")
    
    # Determine if this is code generation or evaluation
    is_evaluation = eval_model is not None and eval_prompt_type is not None
    
    if is_evaluation:
        print(f"Eval Model: {eval_model}")
        print(f"Eval Prompt Type: {eval_prompt_type}")
        print("Processing EVALUATION batch results...")
        # Path: dataset/code_gen_model/eval_model/eval_prompt_type/
        structured_cache_dir = os.path.join(cache_dir, dataset, code_gen_model, eval_model, eval_prompt_type)
    else:
        print("Processing CODE GENERATION batch results...")
        # Path: dataset/code_gen_model/
        structured_cache_dir = os.path.join(cache_dir, dataset, code_gen_model)
    
    os.makedirs(structured_cache_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    if not os.path.exists(results_file):
        print(f"Error: Input file {results_file} does not exist.")
        return

    print(f"Processing results from: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                result = json.loads(line)
                
                # Verify structure
                custom_id = result.get("custom_id")
                
                if not custom_id:
                    print(f"Line {line_num}: Missing 'custom_id'. Skipping.")
                    error_count += 1
                    continue
                
                # Check for errors in the individual response
                if result.get("error"):
                    print(f"Line {line_num}: Request failed for {custom_id}. Error: {result['error']}")
                    error_count += 1
                    continue
                    
                response = result.get("response")
                if not response or not response.get("body"):
                    print(f"Line {line_num}: Missing response body for {custom_id}.")
                    error_count += 1
                    continue
                
                body = response["body"]
                choices = body.get("choices")
                if not choices or len(choices) == 0:
                     print(f"Line {line_num}: No choices in response for {custom_id}.")
                     error_count += 1
                     continue
                     
                content = choices[0].get("message", {}).get("content")
                if content is None:
                    print(f"Line {line_num}: No content in message for {custom_id}.")
                    error_count += 1
                    continue

                # custom_id is the task_id
                task_id = custom_id
                
                # Save to structured cache
                cache_path = os.path.join(structured_cache_dir, f"{task_id}.json")
                with open(cache_path, 'w', encoding='utf-8') as cache_f:
                    json.dump({"content": content}, cache_f)
                
                processed_count += 1
                
            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON.")
                error_count += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error: {e}")
                error_count += 1

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Successfully processed and cached: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Cache location: {structured_cache_dir}")

def parse_filename_metadata(results_file: str):
    """
    Parse metadata from batch results filename.
    
    Code generation format: dataset_MODEL_START_END_results.jsonl
    Example: leetcode_gpt-4o-mini_0_164_results.jsonl
    
    Evaluation format: dataset_code_model_MODEL_PROMPT_eval_eval_EVALMODEL_START_END_results.jsonl
    Example: leetcode_code_model_gpt-4o-mini_vanilla_eval_eval_gpt-4o_0_39_results.jsonl
    
    Returns:
        (dataset, code_gen_model, eval_model, eval_prompt_type)
        For code gen: (dataset, code_gen_model, None, None)
        For eval: (dataset, code_gen_model, eval_model, eval_prompt_type)
    """
    basename = os.path.basename(results_file)
    
    # Try evaluation pattern first (more specific)
    # Pattern: dataset_code_model_CODEMODEL_PROMPTTYPE_eval_eval_EVALMODEL_START_END_results.jsonl
    eval_pattern = r'(.+?)_code_model_(.+?)_(.+?)_eval_eval_(.+?)_\d+_\d+_results\.jsonl'
    match = re.match(eval_pattern, basename)
    
    if match:
        dataset = match.group(1)
        code_gen_model = match.group(2)
        eval_prompt_type = match.group(3)
        eval_model = match.group(4)
        return dataset, code_gen_model, eval_model, eval_prompt_type
    
    # Try code generation pattern
    # Pattern: dataset_MODEL_START_END_results.jsonl
    code_pattern = r'(.+?)_(.+?)_\d+_\d+_results\.jsonl'
    match = re.match(code_pattern, basename)
    
    if match:
        dataset = match.group(1)
        code_gen_model = match.group(2)
        return dataset, code_gen_model, None, None
    
    return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="Process OpenAI Batch API results and populate structured cache.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the results .jsonl file")
    # Optional overrides (rarely needed - only if filename doesn't follow convention)
    parser.add_argument("--dataset", type=str, help="Override dataset name (auto-detected from filename)")
    parser.add_argument("--code_gen_model", type=str, help="Override code generation model (auto-detected from filename)")
    parser.add_argument("--eval_model", type=str, help="Override evaluation model (auto-detected from filename, only for eval results)")
    parser.add_argument("--eval_prompt_type", type=str, help="Override evaluation prompt type (auto-detected from filename, only for eval results)")

    args = parser.parse_args()
    
    # Auto-detect metadata from filename
    print("Auto-detecting metadata from filename...")
    dataset, code_gen_model, eval_model, eval_prompt_type = parse_filename_metadata(args.results_file)
    
    # Use command-line overrides if provided, otherwise use auto-detected values
    final_dataset = args.dataset or dataset
    final_code_gen_model = args.code_gen_model or code_gen_model
    final_eval_model = args.eval_model or eval_model
    final_eval_prompt_type = args.eval_prompt_type or eval_prompt_type
    
    # Validate required arguments
    if not final_dataset or not final_code_gen_model:
        print("Error: Could not determine required metadata from filename.")
        print(f"  Filename: {args.results_file}")
        print(f"  Detected - Dataset: {dataset}, Code Gen Model: {code_gen_model}")
        print("\nExpected filename formats:")
        print("  Code gen: dataset_MODEL_START_END_results.jsonl")
        print("  Eval: dataset_code_model_MODEL_PROMPT_eval_eval_EVALMODEL_START_END_results.jsonl")
        print("\nYou can manually override with --dataset and --code_gen_model arguments.")
        sys.exit(1)
    
    # Determine cache directory
    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    
    process_and_cache_results(
        args.results_file, 
        final_dataset,
        final_code_gen_model,
        final_eval_model,
        final_eval_prompt_type,
        cache_dir
    )

if __name__ == "__main__":
    main()
