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
                
                # Check for HIRE step suffix (e.g., _step_0)
                active_cache_dir = structured_cache_dir
                step_match = re.search(r'_step_(\d+)$', task_id)
                if step_match:
                    step_idx = int(step_match.group(1))
                    # Create a subfolder for this step (1-indexed for readability)
                    active_cache_dir = f"{structured_cache_dir}_step_{step_idx + 1}"
                    os.makedirs(active_cache_dir, exist_ok=True)
                    # Strip step suffix from task_id for the filename
                    task_id = task_id[:step_match.start()]
                
                # Save to structured cache
                cache_path = os.path.join(active_cache_dir, f"{task_id}.json")
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
    
    # KNOWN PROMPTS (from generate_batch.py/utils.prompts)
    # We use these to robustly split the filename.
    # Order matters: more specific (longer) prompts first to avoid partial matches
    KNOWN_PROMPTS = [
        "hire_implementation_checker_context", 
        "hire_implementation_checker_isolated",
        "hire_plan_checker",
        "hire_decomposer",
        "cj_fault_localization",
        "cj_analysis",
        "cj_summary",
        "vanilla"
    ]
    
    # Sort by length descending to match longest first
    KNOWN_PROMPTS.sort(key=len, reverse=True)

    # Try evaluation pattern first (more specific)
    # Pattern 1: dataset_code_model_CODEMODEL_PROMPTTYPE_eval_eval_EVALMODEL_START_END_results.jsonl
    eval_pattern_model = r'(.+?)_code_model_(.+?)_(.+?)_eval_eval_(.+?)_\d+_\d+_results\.jsonl'
    match = re.match(eval_pattern_model, basename)
    
    if match:
        dataset = match.group(1)
        code_gen_model = match.group(2)
        eval_prompt_type = match.group(3)
        eval_model = match.group(4)
        return dataset, code_gen_model, eval_model, eval_prompt_type

    # Pattern 2: dataset_SOURCE_PROMPTTYPE_eval_eval_EVALMODEL_START_END_results.jsonl (for eval_source)
    # Strategy: Iterate through known prompts and check if the filename contains _{prompt}_eval_eval_
    
    KNOWN_DATASETS_UNDERSCORE = ["humaneval_py", "humaneval_java", "humaneval_js", "humaneval_cpp", "humaneval_go"]

    for prompt in KNOWN_PROMPTS:
        # Check for k-variant (e.g. vanilla_k1)
        # Using a regex that optionally matches _k\d+ suffix for the prompt
        # We need to construct a regex dynamically or just check strings.
        # String check is safer.
        
        # Check standard prompt
        anchor = f"_{prompt}_eval_eval_"
        if anchor in basename:
            parts = basename.split(anchor)
            left_part = parts[0] # dataset_SOURCE
            right_part = parts[1] # EVALMODEL_START_END...
            
            # Extract EVALMODEL from right_part
            # right_part looks like: gpt-4o-mini_0_39_results.jsonl
            # match until the last digits
            eval_model_match = re.match(r'(.+?)_\d+_\d+_results\.jsonl', right_part)
            if eval_model_match:
                eval_model = eval_model_match.group(1)
                eval_prompt_type = prompt
                
                # Split left_part into dataset and source
                dataset = None
                code_gen_model = None
                
                # specific check for known datasets with underscore
                for known_ds in KNOWN_DATASETS_UNDERSCORE:
                    if left_part.startswith(known_ds + "_"):
                        dataset = known_ds
                        code_gen_model = left_part[len(known_ds)+1:]
                        break
                
                if not dataset and '_' in left_part:
                    dataset_split = left_part.split('_', 1)
                    dataset = dataset_split[0]
                    code_gen_model = dataset_split[1]
                
                if dataset and code_gen_model:
                     return dataset, code_gen_model, eval_model, eval_prompt_type

        # Check k-variant (e.g. vanilla_k3)
        # Regex for anchor: _{prompt}_k\d+_eval_eval_
        k_anchor_pattern = fr"_{prompt}_k(\d+)_eval_eval_"
        k_match = re.search(fr"{k_anchor_pattern}(.+?)_\d+_\d+_results\.jsonl", basename)
        
        if k_match:
            # We found a match for k-variant
            match_start = k_match.start()
            left_part = basename[:match_start]
            
            # anchor: _vanilla_k1_eval_eval_
            # promt type should be: vanilla_k1
            # k_match.group(1) is the k index (e.g. "1")
            k_val = k_match.group(1)
            eval_prompt_type = f"{prompt}_k{k_val}"
            
            eval_model = k_match.group(2)
            
            # Split left_part into dataset and source
            dataset = None
            code_gen_model = None
            
            for known_ds in KNOWN_DATASETS_UNDERSCORE:
                if left_part.startswith(known_ds + "_"):
                    dataset = known_ds
                    code_gen_model = left_part[len(known_ds)+1:]
                    break
            
            if not dataset and '_' in left_part:
                dataset_split = left_part.split('_', 1)
                dataset = dataset_split[0]
                code_gen_model = dataset_split[1]
            
            if dataset and code_gen_model:
                return dataset, code_gen_model, eval_model, eval_prompt_type

        # Check N-variant (e.g. hire_decomposer_N_3)
        # Regex for anchor: _{prompt}_N\d+_eval_eval_
        
        n_anchor_pattern = fr"_{prompt}_N_(\d+)_eval_eval_"
        n_match = re.search(fr"{n_anchor_pattern}(.+?)_\d+_\d+_results\.jsonl", basename)
        
        if n_match:
             match_start = n_match.start()
             left_part = basename[:match_start]
             
             # n_match.group(1) is the N value (e.g. "3")
             n_val = n_match.group(1)
             eval_prompt_type = f"{prompt}_N_{n_val}"
             
             eval_model = n_match.group(2)
             
             # Split left_part into dataset and source
             dataset = None
             code_gen_model = None
             
             for known_ds in KNOWN_DATASETS_UNDERSCORE:
                if left_part.startswith(known_ds + "_"):
                    dataset = known_ds
                    code_gen_model = left_part[len(known_ds)+1:]
                    break
             
             if not dataset and '_' in left_part:
                dataset_split = left_part.split('_', 1)
                dataset = dataset_split[0]
                code_gen_model = dataset_split[1]
             
             if dataset and code_gen_model:
                return dataset, code_gen_model, eval_model, eval_prompt_type

    # Fallback to generic code generation pattern if no eval pattern matched
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
