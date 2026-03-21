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

def parse_ds_source(ds_source):
    """Split ds_source into dataset and code_gen_model."""
    KNOWN_DATASETS_UNDERSCORE = ["humaneval_py", "humaneval_java", "humaneval_js", "humaneval_cpp", "humaneval_go"]
    dataset = None
    code_gen_model = None
    
    KNOWN_DATASETS_UNDERSCORE.sort(key=len, reverse=True)
    for known_ds in KNOWN_DATASETS_UNDERSCORE:
        if ds_source.startswith(known_ds + "_"):
            dataset = known_ds
            code_gen_model = ds_source[len(known_ds)+1:]
            break
    
    if not dataset and '_' in ds_source:
        dataset_split = ds_source.split('_', 1)
        dataset = dataset_split[0]
        code_gen_model = dataset_split[1]
    
    return dataset, code_gen_model

def parse_filename_metadata(basename: str):
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
    basename = os.path.basename(basename)
    
    code_gen_model = None
    dataset = None
    eval_model = None
    eval_prompt_type = None
    
    # KNOWN PROMPTS (from generate_batch.py/utils.prompts)
    # We use these to robustly split the filename.
    # Order matters: more specific (longer) prompts first to avoid partial matches
    KNOWN_PROMPTS = [
        "hire_explanation_self_refine",
        "hire_explanation_direct_update",
        "hire_explanation_update",
        "hire_explanation_feedback",
        "hire_aggregator_a2_aware_query_aware_flexible",
        "hire_aggregator_a2_aware_flexible",
        "hire_aggregator_query_aware_flexible",
        "hire_aggregator_flexible",
        "hire_explainer_obj_alignment_checker_query_aware_lambda",
        "hire_explainer_obj_alignment_checker_lambda",
        "hire_explainer_obj_alignment_checker_query_aware",
        "hire_explainer_obj_alignment_checker",
        "hire_explainer_obj_checker_query_aware_lambda",
        "hire_explainer_obj_checker_lambda",
        "hire_explainer_obj_query_aware_lambda",
        "hire_explainer_obj_lambda",
        "hire_explainer_obj_checker_query_aware",
        "hire_explainer_obj_checker",
        "hire_explainer_obj_query_aware",
        "hire_explainer_obj",
        "hire_explainer_alignment_checker_query_aware_lambda",
        "hire_explainer_alignment_checker_lambda",
        "hire_explainer_alignment_checker_query_aware",
        "hire_explainer_alignment_checker",
        "hire_explainer_checker_query_aware_lambda",
        "hire_explainer_checker_lambda",
        "hire_explainer_query_aware_lambda",
        "hire_explainer_lambda",
        "hire_explainer_checker_query_aware",
        "hire_explainer_checker",
        "hire_explainer_query_aware",
        "hire_explainer",
        "hire_pseudo_obj_alignment_checker_query_aware_lambda",
        "hire_pseudo_obj_alignment_checker_lambda",
        "hire_pseudo_obj_alignment_checker_query_aware",
        "hire_pseudo_obj_alignment_checker",
        "hire_pseudo_obj_checker_query_aware_lambda",
        "hire_pseudo_obj_checker_lambda",
        "hire_pseudo_obj_query_aware_lambda",
        "hire_pseudo_obj_lambda",
        "hire_pseudo_obj_checker_query_aware",
        "hire_pseudo_obj_checker",
        "hire_pseudo_obj_query_aware",
        "hire_pseudo_obj",
        "hire_pseudo_alignment_checker_query_aware_lambda",
        "hire_pseudo_alignment_checker_lambda",
        "hire_pseudo_alignment_checker_query_aware",
        "hire_pseudo_alignment_checker",
        "hire_pseudo_checker_query_aware_lambda",
        "hire_pseudo_checker_lambda",
        "hire_pseudo_query_aware_lambda",
        "hire_pseudo_lambda",
        "hire_pseudo_checker_query_aware",
        "hire_pseudo_checker",
        "hire_pseudo_query_aware",
        "hire_pseudo",
        "hire_commentor_checker_query_aware_flexible",
        "hire_commentor_checker_flexible",
        "hire_plan_checker_query_aware_text_only_flexible",
        "hire_plan_checker_text_only_flexible",
        "hire_implementation_checker_context_query_aware_flexible",
        "hire_implementation_checker_isolated_query_aware_flexible",
        "hire_plan_checker_query_aware_flexible",
        "hire_implementation_checker_context_flexible",
        "hire_implementation_checker_isolated_flexible",
        "hire_plan_checker_flexible",
        "hire_decomposer_query_aware_flexible",
        "hire_decomposer_flexible",
        "hire_implementation_checker_context_query_aware",
        "hire_implementation_checker_isolated_query_aware",
        "hire_plan_checker_query_aware",
        "hire_decomposer_query_aware",
        "hire_implementation_checker_context", 
        "hire_implementation_checker_isolated",
        "hire_plan_checker",
        "hire_decomposer",
        "cj_fault_localization",
        "cj_analysis",
        "cj_summary",
        "ice_correctness",
        "ice_usefulness",
        "vanilla_no_reasoning",
        "behavior_comparison",
        "behavior_comparison_no_rc",
        "two_phase_reflective",
        "vanilla"
    ]
    
    # Sort by length descending to match longest first
    KNOWN_PROMPTS.sort(key=len, reverse=True)

    # 1. NEW REFINEMENT DETECTION (Prioritize this over evaluation)
    # Format: dataset_SOURCE_PROMPT_eval_eval_EVALMODEL_REFINEMODEL_refine_START_END_results.jsonl
    # Or: dataset_SOURCE_PROMPT_eval_EVALMODEL_REFINEMODEL_refine_START_END_results.jsonl (Legacy)
    refine_match = re.search(r'_([^_]+)_refine_\d+_\d+_results\.jsonl$', basename)
    if refine_match:
        refine_model_name = refine_match.group(1)
        # Part before _refine_: dataset_source_prompt_eval_eval_evalmodel
        remaining = basename[:refine_match.start()]
        
        # We need to find the split point between (dataset_source_prompt) and (eval_model)
        KNOWN_DATASETS_UNDERSCORE = ["humaneval_py", "humaneval_java", "humaneval_js", "humaneval_cpp", "humaneval_go"]
        
        for prompt in KNOWN_PROMPTS:
            # Check for _{prompt}_eval_eval_ or _{prompt}_eval_
            for mid in ["_eval_eval_", "_eval_"]:
                # Allow optional _L\d+, _N_\d+, _k\d+ suffixes for prompt type
                pattern = fr"_{prompt}(_L\d+)?(_N_\d+)?(_k\d+)?{mid}"
                m = re.search(pattern, remaining)
                if m:
                    # Found the split point!
                    ds_source = remaining[:m.start()]
                    # Extract the full prompt type including suffixes
                    eval_prompt_type = remaining[m.start()+1 : m.end()-len(mid)]
                    eval_model = remaining[m.end():]
                    
                    # Split ds_source into dataset and source
                    dataset, code_gen_model = parse_ds_source(ds_source)
                        
                    if dataset and code_gen_model:
                        # Success! Use the specialized format for refinement
                        specialized_gen_model = f"{refine_model_name}_refine_{code_gen_model}_{eval_prompt_type}_{eval_model}"
                        return dataset, specialized_gen_model, None, None

    # 2. Try standard evaluation pattern (Pattern 1)
    # Pattern 1: dataset_code_model_CODEMODEL_PROMPTTYPE_eval_eval_EVALMODEL_START_END_results.jsonl
    eval_pattern_model = r'(.+?)_code_model_(.+?)_(.+?)_eval_eval_(.+?)_\d+_\d+_results\.jsonl'
    match = re.match(eval_pattern_model, basename)
    
    if match:
        dataset = match.group(1)
        code_gen_model = match.group(2)
        eval_prompt_type = match.group(3)
        eval_model = match.group(4)
        return dataset, code_gen_model, eval_model, eval_prompt_type

    # 3. Pattern 2: dataset_SOURCE_PROMPTTYPE_eval_eval_EVALMODEL_START_END_results.jsonl (for eval_source)
    # Strategy: Iterate through known prompts and check if the filename contains _{prompt}_eval_eval_
    
    KNOWN_DATASETS_UNDERSCORE = ["humaneval_py", "humaneval_java", "humaneval_js", "humaneval_cpp", "humaneval_go"]

    for prompt in KNOWN_PROMPTS:
        # Check standard prompt (allowing optional _LX or _faithful/_no_wt suffixes)
        # Suffixes can appear before _lambda or at the end of the prompt part
        lambda_pattern = fr"_{prompt}([a-zA-Z0-9_-]*?)_eval_eval_"
        lambda_match = re.search(lambda_pattern, basename)
        
        if lambda_match:
            anchor = lambda_match.group(0)
            parts = basename.split(anchor)
            left_part = parts[0] # dataset_SOURCE
            right_part = parts[1] # EVALMODEL_START_END...
            
            # Extract EVALMODEL from right_part
            eval_model_match = re.match(r'(.+?)_\d+_\d+_results\.jsonl', right_part)
            if eval_model_match:
                eval_model = eval_model_match.group(1)
                
                # Extract prompt type from anchor 
                eval_prompt_type = anchor[1:-11] # Strip leading _ and trailing _eval_eval_
                
                # Split left_part into dataset and source
                dataset, code_gen_model = parse_ds_source(left_part)
                
                if dataset and code_gen_model:
                     return dataset, code_gen_model, eval_model, eval_prompt_type

        # Check k-variant (e.g. vanilla_k3)
        # Regex for anchor: _{prompt}(_L\d+)?_k\d+_eval_eval_
        k_anchor_pattern = fr"_{prompt}([a-zA-Z0-9_-]*?)_k(\d+)_eval_eval_"
        k_match = re.search(fr"{k_anchor_pattern}(.+?)_\d+_\d+_results\.jsonl", basename)
        
        if k_match:
            # We found a match for k-variant
            match_start = k_match.start()
            left_part = basename[:match_start]
            
            # anchor: _vanilla_k1_eval_eval_
            # promt type should be: vanilla_k1
            suffix_part = k_match.group(1)
            k_val = k_match.group(2)
            eval_prompt_type = f"{prompt}{suffix_part}_k{k_val}"
            
            eval_model = k_match.group(3)
            
            # Split left_part into dataset and source
            dataset, code_gen_model = parse_ds_source(left_part)
            
            if dataset and code_gen_model:
                return dataset, code_gen_model, eval_model, eval_prompt_type

        # Check N-variant (e.g. hire_decomposer_N_3)
        # Regex for anchor: _{prompt}(_L\d+)?_N\d+_eval_eval_
        
        n_anchor_pattern = fr"_{prompt}([a-zA-Z0-9_-]*?)_N_(\d+)_eval_eval_"
        n_match = re.search(fr"{n_anchor_pattern}(.+?)_\d+_\d+_results\.jsonl", basename)
        
        if n_match:
             match_start = n_match.start()
             left_part = basename[:match_start]
             
             # n_match.group(2) is the N value
             suffix_part = n_match.group(1)
             n_val = n_match.group(2)
             eval_prompt_type = f"{prompt}{suffix_part}_N_{n_val}"
             
             eval_model = n_match.group(3)
             
             # Split left_part into dataset and source
             dataset, code_gen_model = parse_ds_source(left_part)
             
             if dataset and code_gen_model:
                return dataset, code_gen_model, eval_model, eval_prompt_type

             if dataset and code_gen_model:
                return dataset, code_gen_model, eval_model, eval_prompt_type

    # 4. Try reconstruction pattern
    # Format: {dataset}_{eval_source}_prompt_suffix_reconstruct_{reconstruct_model}_{start}_{end}_results.jsonl
    reconstruct_match = re.search(r'(.+?)_reconstruct_(.+?)_\d+_\d+_results\.jsonl$', basename)
    if reconstruct_match:
        reconstruct_model = reconstruct_match.group(2)
        # prefix is: {dataset}_{eval_source}_{eval_prompt}{l_suffix}
        prefix = reconstruct_match.group(1)
        
        # We need to split prefix into (dataset_source) and (eval_prompt)
        # Note: eval_prompt here is the prompt that was USED to generate the explanation being reconstructed from.
        for prompt in KNOWN_PROMPTS:
            # Check for _{prompt}_ or ends with _{prompt}
            pattern = fr"_{prompt}(_L\d+)?(_N_\d+)?(_k\d+)?$"
            m = re.search(pattern, prefix)
            if m:
                ds_source = prefix[:m.start()]
                eval_prompt_type = prefix[m.start()+1:] # e.g. hire_explainer_L5
                
                # Split ds_source into dataset and source
                dataset, code_gen_model = parse_ds_source(ds_source)
                
                if dataset and code_gen_model:
                    # We store "reconstruct_{reconstruct_model}" as eval_model and "reconstruct_{eval_prompt}" as prompt type
                    return dataset, code_gen_model, f"reconstruct_{reconstruct_model}", f"reconstruct_{eval_prompt_type}"

    # 5. Try compare pattern
    # Format: {dataset}_{eval_source}_{eval_prompt}{l_suffix}_compare_{model_name_part}_{start}_{end}_results.jsonl
    compare_match = re.search(r'(.+?)_compare_(.+?)_\d+_\d+_results\.jsonl$', basename)
    if compare_match:
        model_name_part = compare_match.group(2)
        prefix = compare_match.group(1)
        for prompt in KNOWN_PROMPTS:
            pattern = fr"_{prompt}(_L\d+)?(_N_\d+)?(_k\d+)?$"
            m = re.search(pattern, prefix)
            if m:
                ds_source = prefix[:m.start()]
                eval_prompt_type = prefix[m.start()+1:]
                dataset, code_gen_model = parse_ds_source(ds_source)
                if dataset and code_gen_model:
                    return dataset, code_gen_model, model_name_part, f"{eval_prompt_type}_compare"

    # 6. Try update pattern
    # Format: {dataset}_{eval_source}_{eval_prompt}{l_suffix}_update_{model_name_part}_{start}_{end}_results.jsonl
    update_match = re.search(r'(.+?)_update_(.+?)_\d+_\d+_results\.jsonl$', basename)
    if update_match:
        model_name_part = update_match.group(2)
        prefix = update_match.group(1)
        for prompt in KNOWN_PROMPTS:
            pattern = fr"_{prompt}(_L\d+)?(_N_\d+)?(_k\d+)?$"
            m = re.search(pattern, prefix)
            if m:
                ds_source = prefix[:m.start()]
                eval_prompt_type = prefix[m.start()+1:]
                dataset, code_gen_model = parse_ds_source(ds_source)
                if dataset and code_gen_model:
                    return dataset, code_gen_model, model_name_part, f"{eval_prompt_type}_update"

    # 7. Try direct_update pattern
    # Format: {dataset}_{eval_source}_{eval_prompt}{l_suffix}_direct_update_{model_name_part}_{start}_{end}_results.jsonl
    direct_update_match = re.search(r'(.+?)_direct_update_(.+?)_\d+_\d+_results\.jsonl$', basename)
    if direct_update_match:
        model_name_part = direct_update_match.group(2)
        prefix = direct_update_match.group(1)
        for prompt in KNOWN_PROMPTS:
            pattern = fr"_{prompt}(_L\d+)?(_N_\d+)?(_k\d+)?$"
            m = re.search(pattern, prefix)
            if m:
                ds_source = prefix[:m.start()]
                eval_prompt_type = prefix[m.start()+1:]
                dataset, code_gen_model = parse_ds_source(ds_source)
                if dataset and code_gen_model:
                    return dataset, code_gen_model, model_name_part, f"{eval_prompt_type}_direct_update"

    # 8. Try self_refine pattern
    # Format: {dataset}_{eval_source}_{eval_prompt}{l_suffix}_self_refine_{model_name_part}_{start}_{end}_results.jsonl
    self_refine_match = re.search(r'(.+?)_self_refine_(.+?)_\d+_\d+_results\.jsonl$', basename)
    if self_refine_match:
        model_name_part = self_refine_match.group(2)
        prefix = self_refine_match.group(1)
        for prompt in KNOWN_PROMPTS:
            pattern = fr"_{prompt}(_L\d+)?(_N_\d+)?(_k\d+)?$"
            m = re.search(pattern, prefix)
            if m:
                ds_source = prefix[:m.start()]
                eval_prompt_type = prefix[m.start()+1:]
                dataset, code_gen_model = parse_ds_source(ds_source)
                if dataset and code_gen_model:
                    return dataset, code_gen_model, model_name_part, f"{eval_prompt_type}_self_refine"

    # 8. Fallback to generic code generation pattern if no eval pattern matched
    code_pattern = r'(.+?)_\d+_\d+_results\.jsonl'
    match = re.match(code_pattern, basename)
    
    if match:
        prefix = match.group(1) # dataset_model
        
        # Split prefix into dataset and source
        dataset, code_gen_model = parse_ds_source(prefix)
        
        if dataset and code_gen_model:
            if code_gen_model.startswith("code_model_"):
                code_gen_model = code_gen_model[len("code_model_"):]
            return dataset, code_gen_model, None, None
    
    # Strip prefix from all result variants before returning
    if code_gen_model and code_gen_model.startswith("code_model_"):
         code_gen_model = code_gen_model[len("code_model_"):]

    return dataset, code_gen_model, eval_model, eval_prompt_type

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
