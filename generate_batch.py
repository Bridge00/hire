import argparse
import json
import os
import sys
import copy
from dotenv import load_dotenv
from openai import OpenAI

from data.all_code_benchmarks import CodeData
#from utils.prompts import CODEGEN_SYS, VANILLA_EVAL_BINARY, CODEJUDGE_ANALYSIS, CODEJUDGE_SUMMARY, HIRE_DECOMPOSER
import utils.prompts as up
from utils.llm import clean_code, _sanitize_filename, is_together_model
from together import Together
from src.eval_class import run_direct_mode

load_dotenv()

# From TextGrad
EVAL_SYS = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise critiques."
EXPLAINER_SYS = "You are a helpful AI assistant that explains code snippets in clear, accurate natural language. Your goal is to be descriptive and objective."
HIRE_PSEUDO_SYS = "You are a critical code evaluator. Your task is to determine if a given pseudocode accurately reflects the logic required by a problem description. You must be rigorous and identify any missing logic or incorrect assumptions in the pseudocode relative to the task requirements."
HIRE_EXPLAINER_SYS = "You are a critical code evaluator. Your task is to determine if a natural language explanation of a code snippet accurately and completely covers the requirements of a problem description. You must ensure the explanation is logically sound and aligns perfectly with the task goals."
HIRE_EXPLAINER_SYS_FAITHFUL = "You are a rigorous code evaluator. Your task is to determine if a natural language explanation of a code snippet accurately and completely covers the requirements of a problem description. **WATCH FOR HALLUCINATIONS**: Do not be fooled by explanations that claim success in walkthroughs while describing flawed logic in the algorithm section. Ensure the described logic AND the walkthroughs are both correct and consistent with the problem."
FEEDBACK_SYS = "You are a critical code reviewer. Your goal is to identify discrepancies between code and its explanation by looking at an implementation reconstructed from that explanation. Provide actionable feedback to improve the explanation."
ARCHITECT_SYS = "You are a Senior Software Architect. Your goal is to write detailed, logical, forward-looking pre-implementation design specifications (blueprints) that describe how to implement target algorithms."
UPDATE_SYS = "You are an expert technical writer. Your goal is to update a code explanation based on feedback and the original source code to ensure perfect accuracy and clarity."

MODEL_MAPPING = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen25-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "llama31-8b": "Bridge00/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-62c81e95",
    "gpt-5.4-thinking": "gpt-5.4",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6"
}

def resolve_model(model_name):
    return MODEL_MAPPING.get(model_name, model_name)



def augment_code_with_comments(original_code, steps):
    """
    Injects comments into original_code based on steps decomposition.
    Uses cursor-based matching to handle partial/fuzzy segments.
    """
    augmented = ""
    cursor = 0
    
    for step in steps:
        expl = step.get("explanation", "").replace("\n", " ")
        seg = step.get("code_segment", "")
        if not seg: continue
        
        # 1. Try exact match
        idx = original_code.find(seg, cursor)
        match_len = len(seg)
        
        # 2. Try first-line match (if exact failed)
        if idx == -1:
            lines = seg.strip().split('\n')
            if lines:
                first_line = lines[0].strip()
                # Simple find for the first line content
                # Note: this finds the first occurrence of the str stripped. 
                # Ideally we want to match it with correct indentation, but text-search is safer relative to cursor.
                candidate = original_code.find(first_line, cursor)
                if candidate != -1:
                    idx = candidate
                    match_len = len(first_line)
        
        if idx != -1:
            # Found match.
            # Append Gap (Unmatched Code between cursor and match)
            augmented += original_code[cursor:idx]
            
            # Insert Comment
            # Ensure newline before comment if not at start?
            if augmented and not augmented.endswith('\n'):
                 augmented += "\n"
            augmented += f"# SUMMARIZER AGENT : {expl}\n"
            
            # Append Matched Part
            augmented += original_code[idx : idx + match_len]
            
            # Advance Cursor
            cursor = idx + match_len
        else:
            # No match found. Insert comment at current cursor?
            # This ensures we don't lose the explanation, even if code alignment failed.
            if augmented and not augmented.endswith('\n'):
                 augmented += "\n"
            augmented += f"# SUMMARIZER AGENT : {expl} (Location Approx)\n"
            
    # Append remaining code
    augmented += original_code[cursor:]
    return augmented

def generate_code_batch(args, dataset, cache_dir):
    """Generate batch requests for code generation."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating CODE GENERATION batch for items {start} to {end}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.code_gen_model}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks and str(task_id) not in args.tasks:
                    skipped_count += 1
                    continue
                
                # Check structured cache: dataset/code_gen_model/task_id.json
                cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.code_gen_model), f"{_sanitize_filename(task_id)}.json")
                
                if not args.force and os.path.exists(cache_path):
                    skipped_count += 1
                    continue
                
                # Determine programming language
                if "language" in row:
                    raw_lang = row["language"].lower()
                    if "python" in raw_lang: programming_language = "python"
                    elif "cpp" in raw_lang or "c++" in raw_lang: programming_language = "cpp"
                    elif "java" in raw_lang: programming_language = "java"
                    elif "js" in raw_lang or "javascript" in raw_lang: programming_language = "javascript"
                    elif "go" in raw_lang: programming_language = "go"
                    else: programming_language = "python"
                elif args.dataset in set(["leetcode", "humaneval_py", "debugbench_hard"]):
                    programming_language = "python"
                elif args.dataset == "humaneval_js":
                    programming_language = "javascript"
                elif args.dataset == "humaneval_java":
                    programming_language = "java"
                elif args.dataset == "humaneval_cpp":
                    programming_language = "c++"
                elif args.dataset == "humaneval_go":
                    programming_language = "go"
                else:
                    assert False, f"Unknown dataset: {args.dataset}"

                model_name = resolve_model(args.code_gen_model)
                is_together = is_together_model(model_name)

                if is_together:
                    request_body = {
                        "custom_id": task_id,
                        "body": {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=programming_language.upper(), 
                                                                PROGRAM_LANGUAGE_LOWER=programming_language)},
                                {"role": "user", "content": prompt}
                            ],
                        }
                    }
                else:
                    request_body = {
                        "custom_id": task_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=programming_language.upper(), 
                                                                PROGRAM_LANGUAGE_LOWER=programming_language)},
                                {"role": "user", "content": prompt}
                            ],
                        }
                    }
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Batch file generated: {output_file}")
    print(f"Total items in range: {end - start}")
    print(f"Requests created: {requests_created}")
    print(f"Skipped (cached): {skipped_count}")
    
    return output_file

def generate_eval_batch(args, dataset, cache_dir, k_index=None, temperature=None):
    """Generate batch requests for evaluation."""
    # GPT-5 models do not support temperature=0.0 (must be 1.0/default)
    if temperature == 0.0 and "gpt-5" in args.eval_model:
        temperature = 1.0
        
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating EVALUATION batch for items {start} to {end} with prompt '{args.eval_prompt}'")
    
    # Determine the system prompt for this batch
    active_system_prompt = EVAL_SYS
    if "explainer" in args.eval_prompt and "checker" not in args.eval_prompt:
        active_system_prompt = EXPLAINER_SYS
    elif "decomposer" in args.eval_prompt or "commentor" in args.eval_prompt:
        active_system_prompt = EXPLAINER_SYS
    elif "hire_pseudo_checker" in args.eval_prompt:
        active_system_prompt = HIRE_PSEUDO_SYS
    elif "hire_explainer_checker" in args.eval_prompt or "hire_explainer_alignment_checker" in args.eval_prompt:
        if "faithful" in args.eval_prompt:
            active_system_prompt = HIRE_EXPLAINER_SYS_FAITHFUL
        else:
            active_system_prompt = HIRE_EXPLAINER_SYS
    elif args.eval_prompt == "imp_plan":
        active_system_prompt = ARCHITECT_SYS

    
    
    if args.eval_source:
        code_source = args.eval_source
        model_or_source = args.eval_source
    else:
        model_or_source = getattr(args, 'code_gen_model', None) or getattr(args, 'explainer_model', None) or args.eval_model
        code_source = f'code_model_{model_or_source}'

    eval_prompt_type = args.eval_prompt
    if args.lambda_val is not None:
        if "_lambda" not in eval_prompt_type:
            eval_prompt_type = f"{args.eval_prompt}_lambda_L{args.lambda_val}"
        else:
            eval_prompt_type = f"{args.eval_prompt}_L{args.lambda_val}"
    elif (args.eval_prompt.startswith("hire_") and 
          "explainer" not in args.eval_prompt and 
          "pseudo" not in args.eval_prompt):
        eval_prompt_type = f"{args.eval_prompt}_N_{args.n}"
        
    if k_index is not None:
        eval_prompt_type = f"{eval_prompt_type}_k{k_index}"

    model_name = f"{args.explainer_model}_{args.eval_model}" if args.explainer_model else args.eval_model

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, 
        f"{args.dataset}_{code_source}_{eval_prompt_type}_eval_eval_{model_name}_{start}_{end}.jsonl"
    )
    
    requests_created = 0
    skipped_count = 0
    missing_code_count = 0
    missing_eval_count = 0
    missing_analysis_count = 0
    missing_plan_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:


        for i in range(start, end):
            try:
                task_id, problem_prompt, _, canonical_solution, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks:
                    if str(task_id) not in args.tasks:
                        skipped_count += 1
                        continue
                
                raw_code = ""

                if args.eval_source:
                    # Access raw item to get arbitrary key
                    raw_item = row #dataset.dataset[i]
                    raw_code = raw_item.get(args.eval_source, "")
                    if not raw_code:
                        print(f"Warning: No code found for source '{args.eval_source}' in task {task_id}")
                        missing_code_count += 1
                        continue
                else:
                    # 1. Locate the generated code in structured cache
                    gen_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.code_gen_model), f"{_sanitize_filename(task_id)}.json")
                    
                    if not os.path.exists(gen_cache_path):
                        missing_code_count += 1
                        continue
                    
                    with open(gen_cache_path, 'r', encoding='utf-8') as f:
                        gen_data = json.load(f)
                        raw_code = gen_data.get("content", "")

                cleaned_code = clean_code(raw_code)

                # 2. Construct Evaluation Prompt
                eval_user_prompt = ""
                
                if args.eval_prompt == "vanilla":
                    eval_user_prompt = up.VANILLA_EVAL_BINARY.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "vanilla"
                elif args.eval_prompt == "imp_plan":
                    eval_user_prompt = up.IMP_PLAN.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "imp_plan"
                elif args.eval_prompt == "vanilla_no_reasoning":
                    eval_user_prompt = up.VANILLA_EVAL_BINARY_NO_REASONING.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "vanilla_no_reasoning"
                elif args.eval_prompt == "behavior_comparison":
                    eval_user_prompt = up.BEHAVIOR_COMPARISON.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "behavior_comparison"
                elif args.eval_prompt == "behavior_comparison_no_rc":
                    eval_user_prompt = up.BEHAVIOR_COMPARISON_NO_RC.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "behavior_comparison_no_rc"
                elif "behavior_comparison_explanation" in args.eval_prompt:
                    # Load explanation from cache
                    is_query_aware = "query_aware" in (args.eval_prompt or "")
                    is_objective = "obj" in (args.eval_prompt or "")
                    source_prompt = "hire_explainer"
                    if is_objective: source_prompt += "_obj"
                    if is_query_aware: source_prompt += "_query_aware"
                    source_folder = source_prompt
                    if args.lambda_val is not None:
                         source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                    
                    source_model = args.explainer_model or args.eval_model
                    expl_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
                    if not os.path.exists(expl_path):
                        missing_eval_count += 1
                        continue
                    with open(expl_path, 'r', encoding='utf-8') as f:
                        explanation = json.load(f).get("content", "")
                    
                    eval_user_prompt = up.BEHAVIOR_COMPARISON_EXPLANATION.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
                    active_eval_prompt_type = args.eval_prompt
                elif args.eval_prompt == "two_phase_reflective":
                    eval_user_prompt = up.TWO_PHASE_REFLECTIVE_EVAL.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "two_phase_reflective"
                elif "two_phase_reflective_explanation" in args.eval_prompt:
                    # Load explanation from cache
                    is_query_aware = "query_aware" in (args.eval_prompt or "")
                    is_objective = "obj" in (args.eval_prompt or "")
                    source_prompt = "hire_explainer"
                    if is_objective: source_prompt += "_obj"
                    if is_query_aware: source_prompt += "_query_aware"
                    source_folder = source_prompt
                    if args.lambda_val is not None:
                         source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                    
                    source_model = args.explainer_model or args.eval_model
                    expl_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
                    if not os.path.exists(expl_path):
                        missing_eval_count += 1
                        continue
                    with open(expl_path, 'r', encoding='utf-8') as f:
                        explanation = json.load(f).get("content", "")
                    
                    eval_user_prompt = up.TWO_PHASE_REFLECTIVE_EVAL_EXPLANATION.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
                    active_eval_prompt_type = args.eval_prompt

                elif "hire_explainer" in args.eval_prompt and "checker" not in args.eval_prompt:
                    base_prompt = args.eval_prompt.replace("_lambda", "")
                    
                    # Handle experimental suffixes
                    suffix = ""
                    if "_faithful" in base_prompt:
                        suffix = "_FAITHFUL"
                        base_prompt = base_prompt.replace("_faithful", "")
                    elif "_no_wt" in base_prompt:
                        suffix = "_NO_WT"
                        base_prompt = base_prompt.replace("_no_wt", "")

                    if args.lambda_val is not None:
                         # Normalize to HIRE_EXPLAINER_[OBJ_]LX[_QUERY_AWARE][SUFFIX]
                         obj_p = "OBJ_" if "explainer_obj" in base_prompt else ""
                         if "query_aware" in base_prompt:
                               prompt_key = f"HIRE_EXPLAINER_{obj_p}L{args.lambda_val}_QUERY_AWARE{suffix}"
                         else:
                               prompt_key = f"HIRE_EXPLAINER_{obj_p}L{args.lambda_val}{suffix}"
                    else:
                         prompt_key = f"{base_prompt.upper()}{suffix}"
                    
                    prompt_template = getattr(up, prompt_key, None)
                    if prompt_template is None:
                         # Fallback to the base version of the lambda prompt
                         base_key = prompt_key.replace(suffix, "") if suffix else prompt_key
                         prompt_template = getattr(up, base_key, None)
                         
                    if prompt_template is None:
                         # Ultimate fallback to generic prompts
                         is_obj = "explainer_obj" in args.eval_prompt
                         if "query_aware" in args.eval_prompt:
                               prompt_template = up.HIRE_EXPLAINER_OBJ_QUERY_AWARE if is_obj else up.HIRE_EXPLAINER_QUERY_AWARE
                         else:
                               prompt_template = up.HIRE_EXPLAINER_OBJ if is_obj else up.HIRE_EXPLAINER
                    
                    # Apply experimental modifications if suffix is set
                    if suffix == "_FAITHFUL":
                        prompt_template = prompt_template.replace(
                            "Describe exactly what the code does",
                            "Describe exactly what the code does. The explanation must be FAITHFUL to the implementation; avoid any 'consistency hallucinations' where you describe what the code SHOULD do instead of what it ACTUALLY does."
                        )
                    elif suffix == "_NO_WT":
                        lines = prompt_template.split('\n')
                        prompt_template = '\n'.join([l for l in lines if "walkthrough" not in l.lower()])
                    
                    if "query_aware" in args.eval_prompt:
                         eval_user_prompt = prompt_template.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    else:
                         eval_user_prompt = prompt_template.format(CODE=cleaned_code)
                    
                    active_eval_prompt_type = args.eval_prompt
                    if args.lambda_val is not None:
                         if "_lambda" not in active_eval_prompt_type:
                             active_eval_prompt_type = f"{args.eval_prompt}_lambda_L{args.lambda_val}"
                         else:
                             active_eval_prompt_type = f"{args.eval_prompt}_L{args.lambda_val}"

                elif "hire_pseudo" in args.eval_prompt and "checker" not in args.eval_prompt:
                    base_prompt = args.eval_prompt.replace("_lambda", "")
                    if args.lambda_val is not None:
                         # Normalize to HIRE_PSEUDO_LX[_QUERY_AWARE]
                         if "query_aware" in base_prompt:
                              prompt_key = f"HIRE_PSEUDO_L{args.lambda_val}_QUERY_AWARE"
                         else:
                              prompt_key = f"HIRE_PSEUDO_L{args.lambda_val}"
                    else:
                         prompt_key = base_prompt.upper()

                    prompt_template = getattr(up, prompt_key, None)
                    if prompt_template is None:
                         prompt_template = up.HIRE_PSEUDO_QUERY_AWARE if "query_aware" in args.eval_prompt else up.HIRE_PSEUDO
                    
                    if "query_aware" in args.eval_prompt or "{PROBLEM}" in prompt_template:
                         eval_user_prompt = prompt_template.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    else:
                         eval_user_prompt = prompt_template.format(CODE=cleaned_code)
                    
                    active_eval_prompt_type = args.eval_prompt
                    if args.lambda_val is not None:
                         if "_lambda" not in active_eval_prompt_type:
                             active_eval_prompt_type = f"{args.eval_prompt}_lambda_L{args.lambda_val}"
                         else:
                             active_eval_prompt_type = f"{args.eval_prompt}_L{args.lambda_val}"

                elif "hire_pseudo" in args.eval_prompt and "checker" in args.eval_prompt:
                    is_query_aware = "query_aware" in args.eval_prompt
                    source_prompt = "hire_pseudo_query_aware" if is_query_aware else "hire_pseudo"
                    source_folder = source_prompt
                    if args.lambda_val is not None:
                         source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                    
                    # Check structured cache for the pseudocode
                    source_model = args.explainer_model or args.eval_model
                    pseudo_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
                    
                    if not os.path.exists(pseudo_path):
                        missing_eval_count += 1
                        continue
                        
                    with open(pseudo_path, 'r', encoding='utf-8') as f:
                        pseudo_data = json.load(f)
                        pseudocode = pseudo_data.get("content", "")
                        
                    eval_user_prompt = up.HIRE_PSEUDO_CHECKER.format(PROBLEM=problem_prompt, PSEUDOCODE=pseudocode)
                    active_eval_prompt_type = args.eval_prompt

                elif "hire_explainer" in args.eval_prompt and "checker" in args.eval_prompt and "alignment" not in args.eval_prompt:
                    is_query_aware = "query_aware" in args.eval_prompt
                    is_objective = "explainer_obj" in args.eval_prompt
                    
                    source_prompt = "hire_explainer"
                    if is_objective: source_prompt += "_obj"
                    if is_query_aware: source_prompt += "_query_aware"
                    
                    source_folder = source_prompt
                    if args.lambda_val is not None:
                         source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                    
                    # Check structured cache for the explanation
                    source_model = args.explainer_model or args.eval_model
                    explainer_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
                    
                    if not os.path.exists(explainer_path):
                        missing_eval_count += 1
                        continue
                        
                    with open(explainer_path, 'r', encoding='utf-8') as f:
                        expl_data = json.load(f)
                        explanation = expl_data.get("content", "")
                        
                    eval_user_prompt = up.HIRE_EXPLAINER_CHECKER.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
                    active_eval_prompt_type = args.eval_prompt

                elif "hire_explainer" in args.eval_prompt and "alignment_checker" in args.eval_prompt:
                    #print('hire_explainer and alignment_checker')
                    is_query_aware = "query_aware" in args.eval_prompt
                    is_objective = "explainer_obj" in args.eval_prompt
                    is_faithful = "faithful" in args.eval_prompt
                    is_no_wt = "no_wt" in args.eval_prompt
                    is_code_checker = "alignment_checker_code" in args.eval_prompt

                    source_prompt = "hire_explainer"
                    if is_objective: source_prompt += "_obj"
                    if is_query_aware: source_prompt += "_query_aware"
                    
                    suffix = ""
                    # For the alignment checker, we want to check the specific experimental explanation
                    if is_faithful: suffix = "_faithful"
                    elif is_no_wt: suffix = "_no_wt"
                    
                    update_suffix = ""
                    if "direct_update" in args.eval_prompt: update_suffix = "_direct_update"
                    elif "self_refine" in args.eval_prompt: update_suffix = "_self_refine"
                    elif "update" in args.eval_prompt: update_suffix = "_update"
                    elif "style_transfer" in args.eval_prompt: update_suffix = "_style_transfer"
                    
                    # Determine the source folder for the initial or updated explanation
                    source_folder = source_prompt + suffix + update_suffix
                    if args.lambda_val is not None:
                         # Handle naming mismatch: initial explains use _lambda_L, while updates use _L
                         if update_suffix:
                              source_folder = f"{source_prompt}{suffix}_L{args.lambda_val}{update_suffix}"
                         else:
                              source_folder = f"{source_prompt}{suffix}_lambda_L{args.lambda_val}"
                         
                         # Robust check: if one doesn't exist, try the other
                         explainer_model = args.explainer_model or args.eval_model
                         if update_suffix == "_style_transfer":
                              style_transfer_model = args.style_transfer_model or explainer_model
                              source_model = f"{explainer_model}_{style_transfer_model}" if style_transfer_model != explainer_model else explainer_model
                         else:
                              source_model = explainer_model
                         check_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
                         if not os.path.exists(check_path):
                              alt_folder = f"{source_prompt}{suffix}_lambda_L{args.lambda_val}{update_suffix}" if update_suffix else f"{source_prompt}{suffix}_L{args.lambda_val}"
                              alt_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, alt_folder, f"{_sanitize_filename(task_id)}.json")
                              if os.path.exists(alt_path):
                                   source_folder = alt_folder
                    
                    # Check structured cache for the explanation
                    explainer_model = args.explainer_model or args.eval_model
                    if update_suffix == "_style_transfer":
                        style_transfer_model = args.style_transfer_model or explainer_model
                        source_model = f"{explainer_model}_{style_transfer_model}" if style_transfer_model != explainer_model else explainer_model
                    else:
                        source_model = explainer_model
                    explainer_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
                    
                    if not os.path.exists(explainer_path):
                        missing_eval_count += 1
                        continue
                        
                    with open(explainer_path, 'r', encoding='utf-8') as f:
                        expl_data = json.load(f)
                        explanation = expl_data.get("content", "")
                    
                    if is_code_checker:
                        #print('is code checker')
                        checker_prompt = up.HIRE_EXPLAINER_ALIGNMENT_CHECKER_CODE
                    elif is_faithful:
                        checker_prompt = up.HIRE_EXPLAINER_ALIGNMENT_CHECKER_FAITHFUL
                    else:
                        checker_prompt = up.HIRE_EXPLAINER_ALIGNMENT_CHECKER
                        
                    eval_user_prompt = checker_prompt.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
                    active_eval_prompt_type = args.eval_prompt

                # HIRE Family Logic
                elif args.eval_prompt.startswith("hire_"):
                    # Determine query awareness
                    is_query_aware_decomposer = "hire_decomposer_query_aware" in args.eval_prompt
                    
                    # Decomposer Prompt
                    # Decomposer Prompt
                    if "hire_decomposer_flexible" in args.eval_prompt:
                         if is_query_aware_decomposer:
                              eval_user_prompt = up.HIRE_DECOMPOSER_WITH_PROBLEM_AT_MOST_N.format(PROBLEM=problem_prompt, N=args.n, CODE=cleaned_code)
                              active_eval_prompt_type = f"hire_decomposer_query_aware_flexible_N_{args.n}"
                         else:
                              eval_user_prompt = up.HIRE_DECOMPOSER_AT_MOST_N.format(N=args.n, CODE=cleaned_code)
                              active_eval_prompt_type = f"hire_decomposer_flexible_N_{args.n}"

                    elif "hire_decomposer" in args.eval_prompt:
                         if is_query_aware_decomposer:
                             eval_user_prompt = up.HIRE_DECOMPOSER_WITH_PROBLEM.format(PROBLEM=problem_prompt, N=args.n, CODE=cleaned_code)
                             active_eval_prompt_type = f"hire_decomposer_query_aware_N_{args.n}"
                         else:
                             eval_user_prompt = up.HIRE_DECOMPOSER.format(N=args.n, CODE=cleaned_code)
                             active_eval_prompt_type = f"hire_decomposer_N_{args.n}"
                             
                    # Downstream Checkers (Plan, Impl)
                    else:
                        # Determine source decomposer folder
                        # Determine source decomposer folder
                        # Check if prompt ends with _query_aware
                        is_query_aware_pipeline = "query_aware" in args.eval_prompt
                        is_flexible_pipeline = "flexible" in args.eval_prompt
                        
                        decomposer_folder_name = "hire_decomposer"
                        if is_query_aware_pipeline:
                            decomposer_folder_name += "_query_aware"
                        if is_flexible_pipeline:
                            decomposer_folder_name += "_flexible"
                            
                        source_decomposer_base = f"{decomposer_folder_name}_N_{args.n}"
                        
                        source_model = args.explainer_model or args.eval_model
                        decomposed_plan_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_decomposer_base), f"{_sanitize_filename(task_id)}.json")
                        
                        if not os.path.exists(decomposed_plan_path):
                            missing_plan_count += 1
                            continue
                        
                        with open(decomposed_plan_path, 'r', encoding='utf-8') as f:
                            plan_data = json.load(f)
                            
                        # PLAN CHECKER
                        if "hire_plan_checker" in args.eval_prompt:
                             plan = plan_data.get("content", "")
                             
                             if "text_only" in args.eval_prompt:
                                 try:
                                     json_start = plan.find('{')
                                     json_end = plan.rfind('}') + 1
                                     plan_json = json.loads(plan[json_start:json_end])
                                     
                                     steps = plan_json.get("steps", [])
                                     text_steps = []
                                     for i, step in enumerate(steps):
                                         # Create a simplified step object with just explanation
                                         text_steps.append({
                                             "step": i + 1,
                                             "explanation": step.get("explanation", "")
                                         })
                                         
                                     plan = json.dumps({"steps": text_steps}, indent=2)
                                 except Exception as e:
                                     print(f"Warning: Failed to strip code from plan for {task_id}: {e}")
                                     # Convert to single string or keep original? 
                                     # If parsing fails, we usually can't verify, so maybe keep original 
                                     # or skip. For now, we will proceed with potentially broken plan 
                                     # but the user should know.
                                     pass

                             eval_user_prompt = up.HIRE_PLAN_CHECKER.format(PROBLEM=problem_prompt, PLAN=plan)
                             active_eval_prompt_type = f"{args.eval_prompt}_N_{args.n}" # e.g. hire_plan_checker_query_aware_N_3
                        
                        elif "hire_commentor_checker" in args.eval_prompt:
                             plan = plan_data.get("content", "")
                             
                             # Construct Augmented Code from Plan
                             try:
                                 json_start = plan.find('{')
                                 json_end = plan.rfind('}') + 1
                                 plan_json = json.loads(plan[json_start:json_end])
                                 steps = plan_json.get("steps", [])
                                 
                                 # Use robust reconstruction using ORIGINAL cleaned_code
                                 augmented_code = augment_code_with_comments(cleaned_code, steps)
                                     
                             except Exception as e:
                                 print(f"Error constructing augmented code for {task_id}: {e}")
                                 continue

                             eval_user_prompt = up.HIRE_COMMENTOR_CODE_CHECKER.format(PROBLEM=problem_prompt, AUGMENTED_CODE=augmented_code)
                             active_eval_prompt_type = f"{args.eval_prompt}_N_{args.n}"
                        
                        # AGGREGATOR
                        elif "hire_aggregator" in args.eval_prompt:
                             # Determine dependencies based on current prompt flags
                             is_query_aware = "query_aware" in args.eval_prompt
                             # Assuming flexible, as this is the new pipeline
                             
                             # Dependency 1: Plan Checker (Text Only)
                             plan_suffix = "_query_aware_text_only_flexible" if is_query_aware else "_text_only_flexible"
                             plan_folder = f"hire_plan_checker{plan_suffix}_N_{args.n}"
                             source_model = args.explainer_model or args.eval_model
                             plan_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(plan_folder), f"{_sanitize_filename(task_id)}.json")
                             
                             # Dependency 2: Commentor
                             commentor_suffix = "_query_aware_flexible" if is_query_aware else "_flexible"
                             commentor_folder = f"hire_commentor_checker{commentor_suffix}_N_{args.n}"
                             commentor_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, commentor_folder, f"{_sanitize_filename(task_id)}.json")
                             
                             if not os.path.exists(plan_path) or not os.path.exists(commentor_path):
                                 # missing_dependency_count?
                                 missing_eval_count += 1
                                 continue
                                 
                             # Load and Extract Reasoning
                             try:
                                 with open(plan_path, 'r', encoding='utf-8') as f:
                                     pd = json.load(f)
                                     pc_content = json.loads(pd.get("content", "{}"))
                                     plan_reasoning = pc_content.get("reasoning", "No reasoning provided.")
                                 
                                 with open(commentor_path, 'r', encoding='utf-8') as f:
                                     cd = json.load(f)
                                     cc_content = json.loads(cd.get("content", "{}"))
                                     commentor_reasoning = cc_content.get("reasoning", "No reasoning provided.")
                             except Exception as e:
                                 print(f"Error parsing reasoning for {task_id}: {e}")
                                 continue
                             
                             prompt_template = up.HIRE_AGGREGATOR_A2_AWARE if "a2_aware" in args.eval_prompt else up.HIRE_AGGREGATOR
                             
                             eval_user_prompt = prompt_template.format(
                                 PROBLEM=problem_prompt,
                                 CODE=cleaned_code,
                                 PLAN_REASONING=plan_reasoning,
                                 COMMENTOR_REASONING=commentor_reasoning
                             )
                             active_eval_prompt_type = f"{args.eval_prompt}_N_{args.n}"

                        # IMPLEMENTATION CHECKERS
                        elif "hire_implementation_checker" in args.eval_prompt:
                             plan_content = plan_data.get("content", "")
                             try:
                                json_start = plan_content.find('{')
                                json_end = plan_content.rfind('}') + 1
                                plan_json = json.loads(plan_content[json_start:json_end])
                                steps = plan_json.get("steps", [])
                             except Exception as e:
                                print(f"Error parsing plan JSON for {task_id}: {e}")
                                continue
                                
                             previous_steps_context = ""
                             active_eval_prompt_type = f"{args.eval_prompt}_N_{args.n}"

                             for idx, step in enumerate(steps):
                                step_desc = step.get("explanation", "")
                                step_code = step.get("code_segment", "")
                                step_task_id = f"{task_id}_step_{idx}"
                                
                                # isolated vs context checker logic
                                # IMPORTANT: The prompt name might be hire_implementation_checker_isolated_query_aware
                                # so checking "isolated" in prompt string is still valid.
                                
                                checker_logic = "isolated" if "isolated" in args.eval_prompt else "context"
                                
                                if checker_logic == "isolated":
                                    eval_user_prompt = up.HIRE_IMPLEMENTATION_CHECKER_ISOLATED.format(
                                        STEP_DESC=step_desc,
                                        STEP_CODE=step_code
                                    )
                                else: # checker_logic == "context"
                                    eval_user_prompt = up.HIRE_IMPLEMENTATION_CHECKER_CONTEXT.format(
                                        PROBLEM=problem_prompt,
                                        PREVIOUS_STEPS=previous_steps_context if previous_steps_context else "None",
                                        CURRENT_STEP_DESC=step_desc,
                                        CURRENT_STEP_CODE=step_code
                                    )
                                    # Update context
                                    previous_steps_context += f"Step {idx+1}: {step_desc}\nImplementation:\n{step_code}\n\n"

                                # Check cache for step
                                step_folder = f"{active_eval_prompt_type}_step_{idx + 1}"
                                source_model = args.explainer_model or args.eval_model
                                step_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(step_folder), f"{_sanitize_filename(task_id)}.json")
                                if not args.force and os.path.exists(step_cache_path):
                                    skipped_count += 1
                                    continue

                                request_body = {
                                    "custom_id": step_task_id,
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": resolve_model(args.eval_model),
                                        "messages": [
                                            {"role": "system", "content": EVAL_SYS},
                                            {"role": "user", "content": eval_user_prompt}
                                        ],
                                    }
                                }
                                if temperature is not None:
                                    request_body["body"]["temperature"] = temperature
                                
                                f_out.write(json.dumps(request_body) + "\n")
                                requests_created += 1
                             
                             continue # Skip single-request write below

                elif args.eval_prompt == "cj_analysis":
                    eval_user_prompt = up.CODEJUDGE_ANALYSIS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "cj_analysis"
                elif args.eval_prompt == "cj_summary":
                    # Check structured cache for analysis
                    analysis_model = args.analysis_model or args.explainer_model or args.eval_model
                    analysis_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(analysis_model), "cj_analysis", f"{_sanitize_filename(task_id)}.json")
                    
                    if not os.path.exists(analysis_path):
                        missing_analysis_count += 1
                        continue
                        
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                        analysis_content = analysis_data.get("content", "")
                        
                    eval_user_prompt = up.CODEJUDGE_SUMMARY.format(ANALYSIS=analysis_content)
                    active_eval_prompt_type = "cj_summary"
                elif args.eval_prompt == "cj_fault_localization":
                    eval_user_prompt = up.CODEJUDGE_FAULT_LOCALIZATION.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "cj_fault_localization"
                elif args.eval_prompt == "ice_correctness":
                    eval_user_prompt = up.ICE_CORRECTNESS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "ice_correctness"
                elif args.eval_prompt == "ice_usefulness":
                    eval_user_prompt = up.ICE_USEFULNESS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "ice_usefulness"

                if not eval_user_prompt:
                    print(f"Error: Empty prompt for item {i}")
                    continue

                # 3. Check if Evaluation is already cached
                cache_prompt_type = active_eval_prompt_type
                if k_index is not None:
                    cache_prompt_type = f"{active_eval_prompt_type}_k{k_index}"

                if args.eval_source:
                        # Cache under 'source_name' instead of 'canonical' vs 'model' dichotomy?
                        # Or just use the source name as the model/source directory component
                        eval_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.eval_source), _sanitize_filename(model_name), _sanitize_filename(cache_prompt_type), f"{_sanitize_filename(task_id)}.json")
                else:
                        eval_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.code_gen_model), _sanitize_filename(model_name), _sanitize_filename(cache_prompt_type), f"{_sanitize_filename(task_id)}.json")

                if not args.force and os.path.exists(eval_cache_path):
                    skipped_count += 1
                    continue

                eval_model_name = resolve_model(args.eval_model)
                is_together = is_together_model(eval_model_name)

                if is_together:
                    request_body = {
                        "custom_id": task_id,
                        "body": {
                            "model": eval_model_name,
                            "messages": [
                                {"role": "system", "content": active_system_prompt},
                                {"role": "user", "content": eval_user_prompt}
                            ],
                        }
                    }
                else:
                    request_body = {
                        "custom_id": task_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": eval_model_name,
                            "messages": [
                                {"role": "system", "content": active_system_prompt},
                                {"role": "user", "content": eval_user_prompt}
                            ],
                        }
                    }
                if temperature is not None:
                    request_body["body"]["temperature"] = temperature
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Total items in range: {end - start}")
    print(f"Requests created: {requests_created}")
    print(f"Skipped (already cached): {skipped_count}")
    print(f"Skipped (missing generated code): {missing_code_count}")
    if args.eval_prompt == "cj_summary":
        print(f"Skipped (missing analysis): {missing_analysis_count}")
    if missing_plan_count > 0 or missing_eval_count > 0:
         print(f"Skipped (missing dependencies/explanations): {missing_plan_count + missing_eval_count}")

    if requests_created == 0:
        print(f"No requests created. Deleting empty file: {output_file}")
        try:
            os.remove(output_file)
        except OSError:
            pass
        return None
    
    print(f"Batch file generated: {output_file}")
    return output_file

def generate_reconstruct_batch(args, dataset, cache_dir):
    """Generate batch requests for code reconstruction from HIRE explanations."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating RECONSTRUCTION batch for items {start} to {end}")
    
    reconstruct_model = args.reconstruct_model or args.eval_model or "gpt-4o-mini"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    l_suffix = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
    model_name_part = f"{args.explainer_model}_{reconstruct_model}" if args.explainer_model else reconstruct_model
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source or 'code'}_{args.eval_prompt}{l_suffix}_reconstruct_{model_name_part}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_explanation_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks and str(task_id) not in args.tasks:
                    skipped_count += 1
                    continue

                # 1. Load explanation from structured cache
                is_query_aware = "query_aware" in (args.eval_prompt or "")
                is_objective = "explainer_obj" in (args.eval_prompt or "")
                
                if args.eval_prompt == "imp_plan":
                    source_prompt = "imp_plan"
                else:
                    source_prompt = "hire_explainer"
                    if is_objective: source_prompt += "_obj"
                    if is_query_aware: source_prompt += "_query_aware"
                
                source_folder = source_prompt
                if args.lambda_val is not None:
                     source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                
                # Use current code source context (canonical vs model)
                if args.eval_source:
                    model_or_source = args.eval_source
                else:
                    model_or_source = args.code_gen_model
                
                source_model = args.explainer_model or args.eval_model
                explainer_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
                
                if not os.path.exists(explainer_path):
                    missing_explanation_count += 1
                    continue
                    
                with open(explainer_path, 'r', encoding='utf-8') as f:
                    expl_data = json.load(f)
                    explanation = expl_data.get("content", "")

                # 2. Construct Reconstruction Prompt
                if args.eval_prompt == "imp_plan":
                    reconstruct_user_prompt = up.IMP_PLAN_RECONSTRUCT.format(PROBLEM=problem_prompt, DESIGN_SPECIFICATION=explanation)
                elif is_query_aware:
                    reconstruct_user_prompt = up.HIRE_RECONSTRUCT_QUERY_AWARE.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
                else:
                    reconstruct_user_prompt = up.HIRE_RECONSTRUCT.format(EXPLANATION=explanation)

                # 3. Determine programming language for system prompt
                if "language" in row:
                    raw_lang = row["language"].lower()
                    if "python" in raw_lang: programming_language = "python"
                    elif "cpp" in raw_lang or "c++" in raw_lang: programming_language = "cpp"
                    elif "java" in raw_lang: programming_language = "java"
                    elif "js" in raw_lang or "javascript" in raw_lang: programming_language = "javascript"
                    elif "go" in raw_lang: programming_language = "go"
                    else: programming_language = "python"
                elif args.dataset in set(["leetcode", "humaneval_py", "debugbench_hard"]):
                    programming_language = "python"
                elif args.dataset == "humaneval_js":
                    programming_language = "javascript"
                elif args.dataset == "humaneval_java":
                    programming_language = "java"
                elif args.dataset == "humaneval_cpp":
                    programming_language = "c++"
                elif args.dataset == "humaneval_go":
                    programming_language = "go"
                else:
                    programming_language = "python" # fallback

                system_prompt = up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=programming_language.upper(), 
                                                   PROGRAM_LANGUAGE_LOWER=programming_language)

                # 4. Create Batch Request
                model_name = resolve_model(reconstruct_model)
                is_together = is_together_model(model_name)

                if is_together:
                    request_body = {
                        "custom_id": task_id,
                        "body": {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": reconstruct_user_prompt}
                            ],
                        }
                    }
                else:
                    request_body = {
                        "custom_id": task_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": reconstruct_user_prompt}
                            ],
                        }
                    }
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Skipped (missing explanations): {missing_explanation_count}")
    
    if requests_created == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
        
    return output_file

def generate_compare_batch(args, dataset, cache_dir):
    """Generate batch requests for explanation feedback by comparing original, explanation, and reconstruction."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating COMPARE batch for items {start} to {end}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    compare_model = args.compare_model or args.eval_model
    l_suffix = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
    model_name_part = f"{args.explainer_model}_{compare_model}" if args.explainer_model else compare_model
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source or 'code'}_{args.eval_prompt}{l_suffix}_compare_{model_name_part}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_data_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks and str(task_id) not in args.tasks:
                    skipped_count += 1
                    continue

                # 1. Load original code
                if args.eval_source:
                    model_or_source = args.eval_source
                    raw_code = row.get(args.eval_source, "")
                else:
                    model_or_source = args.code_gen_model
                    gen_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.code_gen_model), f"{_sanitize_filename(task_id)}.json")
                    if os.path.exists(gen_cache_path):
                        with open(gen_cache_path, 'r', encoding='utf-8') as f:
                            raw_code = json.load(f).get("content", "")
                    else:
                        missing_data_count += 1
                        continue
                
                original_code = clean_code(raw_code)

                # 2. Load explanation
                is_query_aware = "query_aware" in (args.eval_prompt or "")
                is_objective = "explainer_obj" in (args.eval_prompt or "")
                source_prompt = "hire_explainer"
                if is_objective: source_prompt += "_obj"
                if is_query_aware: source_prompt += "_query_aware"
                source_folder = source_prompt
                if args.lambda_val is not None:
                     source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                
                source_model = args.explainer_model or args.eval_model
                explainer_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
                
                if not os.path.exists(explainer_path):
                    missing_data_count += 1
                    continue
                with open(explainer_path, 'r', encoding='utf-8') as f:
                    explanation = json.load(f).get("content", "")

                # 3. Load reconstructed code
                reconstruct_model = args.reconstruct_model or args.code_gen_model or args.eval_model or "gpt-4o-mini"
                l_part = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
                reconstruct_prompt = f"reconstruct_{args.eval_prompt}{l_part}"
                reproduce_model_folder = f"reconstruct_{reconstruct_model}"
                
                reconstruct_path = os.path.join(cache_dir, args.dataset, model_or_source, reproduce_model_folder, reconstruct_prompt, f"{_sanitize_filename(task_id)}.json")
                
                if not os.path.exists(reconstruct_path):
                    import re
                    alt_prompt = re.sub(r'_L(\d+)', r'_lambda_L\1', reconstruct_prompt)
                    reconstruct_path = os.path.join(cache_dir, args.dataset, model_or_source, reproduce_model_folder, alt_prompt, f"{_sanitize_filename(task_id)}.json")
                
                if not os.path.exists(reconstruct_path):
                    missing_data_count += 1
                    continue
                with open(reconstruct_path, 'r', encoding='utf-8') as f:
                    reconstructed_code = json.load(f).get("content", "")

                # 4. Construct Prompt
                compare_prompt = up.HIRE_EXPLANATION_FEEDBACK.format(
                    ORIGINAL_CODE=original_code,
                    EXPLANATION=explanation,
                    RECONSTRUCTED_CODE=reconstructed_code
                )

                # 5. Create Batch Request
                model_name = resolve_model(compare_model)
                is_together = is_together_model(model_name)
                
                request_body = {
                    "custom_id": task_id,
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": FEEDBACK_SYS},
                            {"role": "user", "content": compare_prompt}
                        ],
                    }
                }
                if not is_together:
                    request_body["method"] = "POST"
                    request_body["url"] = "/v1/chat/completions"
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Batch file generated: {output_file}")
    print(f"Requests created: {requests_created}")
    print(f"Skipped: {skipped_count}")
    print(f"Missing data: {missing_data_count}")
    
    if requests_created == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
        
    return output_file

def generate_update_batch(args, dataset, cache_dir):
    """Generate batch requests for updated explanations based on feedback."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating UPDATE batch for items {start} to {end}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    update_model = args.update_model or args.eval_model
    l_suffix = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
    model_name_part = f"{args.explainer_model}_{update_model}" if args.explainer_model else update_model
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source or 'code'}_{args.eval_prompt}{l_suffix}_update_{model_name_part}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_data_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks and str(task_id) not in args.tasks:
                    skipped_count += 1
                    continue

                # 1. Load original code
                if args.eval_source:
                    model_or_source = args.eval_source
                    raw_code = row.get(args.eval_source, "")
                else:
                    model_or_source = args.code_gen_model
                    gen_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.code_gen_model), f"{_sanitize_filename(task_id)}.json")
                    if os.path.exists(gen_cache_path):
                        with open(gen_cache_path, 'r', encoding='utf-8') as f:
                            raw_code = json.load(f).get("content", "")
                    else:
                        missing_data_count += 1
                        continue
                original_code = clean_code(raw_code)

                # 2. Load initial explanation
                is_query_aware = "query_aware" in (args.eval_prompt or "")
                is_objective = "explainer_obj" in (args.eval_prompt or "")
                source_prompt = "hire_explainer"
                if is_objective: source_prompt += "_obj"
                if is_query_aware: source_prompt += "_query_aware"
                source_folder = source_prompt
                if args.lambda_val is not None:
                     source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                
                source_model = args.explainer_model or args.eval_model
                explainer_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
                if not os.path.exists(explainer_path):
                    missing_data_count += 1
                    continue
                with open(explainer_path, 'r', encoding='utf-8') as f:
                    explanation = json.load(f).get("content", "")

                # 3. Load feedback
                compare_prompt_type = f"{args.eval_prompt}{l_suffix}_compare"
                feedback_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(compare_prompt_type), f"{_sanitize_filename(task_id)}.json")
                if not os.path.exists(feedback_path):
                    import re
                    alt_compare_prompt = re.sub(r'_L(\d+)', r'_lambda_L\1', compare_prompt_type)
                    feedback_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(model_or_source), _sanitize_filename(source_model), _sanitize_filename(alt_compare_prompt), f"{_sanitize_filename(task_id)}.json")
                if not os.path.exists(feedback_path):
                    missing_data_count += 1
                    continue
                with open(feedback_path, 'r', encoding='utf-8') as f:
                    feedback = json.load(f).get("content", "")

                # 4. Construct Prompt
                update_user_prompt = up.HIRE_UPDATE_EXPLANATION.format(
                    PROBLEM=problem_prompt,
                    ORIGINAL_CODE=original_code,
                    EXPLANATION=explanation,
                    FEEDBACK=feedback
                )

                # 5. Create Batch Request
                model_name = resolve_model(update_model)
                is_together = is_together_model(model_name)
                
                request_body = {
                    "custom_id": task_id,
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": UPDATE_SYS},
                            {"role": "user", "content": update_user_prompt}
                        ],
                    }
                }
                if not is_together:
                    request_body["method"] = "POST"
                    request_body["url"] = "/v1/chat/completions"
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Batch file generated: {output_file}")
    print(f"Requests created: {requests_created}")
    print(f"Skipped: {skipped_count}")
    print(f"Missing data: {missing_data_count}")
    
    if requests_created == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
        
    return output_file

def generate_direct_update_batch(args, dataset, cache_dir):
    """Generate batch requests for explanation improvement by comparing original and reconstruction directly."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating DIRECT_UPDATE batch for items {start} to {end}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    update_model = args.update_model or args.eval_model
    l_suffix = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
    model_name_part = f"{args.explainer_model}_{update_model}" if args.explainer_model else update_model
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source or 'code'}_{args.eval_prompt}{l_suffix}_direct_update_{model_name_part}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_data_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks and str(task_id) not in args.tasks:
                    skipped_count += 1
                    continue

                # 1. Load original code
                if args.eval_source:
                    model_or_source = args.eval_source
                    raw_code = row.get(args.eval_source, "")
                else:
                    model_or_source = args.code_gen_model
                    gen_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.code_gen_model), f"{_sanitize_filename(task_id)}.json")
                    if os.path.exists(gen_cache_path):
                        with open(gen_cache_path, 'r', encoding='utf-8') as f:
                            raw_code = json.load(f).get("content", "")
                    else:
                        missing_data_count += 1
                        continue
                original_code = clean_code(raw_code)

                # 2. Load initial explanation
                is_query_aware = "query_aware" in (args.eval_prompt or "")
                is_objective = "explainer_obj" in (args.eval_prompt or "")
                source_prompt = "hire_explainer"
                if is_objective: source_prompt += "_obj"
                if is_query_aware: source_prompt += "_query_aware"
                source_folder = source_prompt
                if args.lambda_val is not None:
                     source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                
                source_model = args.explainer_model or args.eval_model
                explainer_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
                if not os.path.exists(explainer_path):
                    missing_data_count += 1
                    continue
                with open(explainer_path, 'r', encoding='utf-8') as f:
                    explanation = json.load(f).get("content", "")

                # 3. Load reconstructed code
                reconstruct_model = args.reconstruct_model or args.code_gen_model or args.eval_model or "gpt-4o-mini"
                l_part = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
                reconstruct_prompt = f"reconstruct_{args.eval_prompt}{l_part}"
                reproduce_model_folder = f"reconstruct_{reconstruct_model}"
                
                reconstruct_path = os.path.join(cache_dir, args.dataset, model_or_source, reproduce_model_folder, reconstruct_prompt, f"{_sanitize_filename(task_id)}.json")
                if not os.path.exists(reconstruct_path):
                    import re
                    alt_prompt = re.sub(r'_L(\d+)', r'_lambda_L\1', reconstruct_prompt)
                    reconstruct_path = os.path.join(cache_dir, args.dataset, model_or_source, reproduce_model_folder, alt_prompt, f"{_sanitize_filename(task_id)}.json")
                if not os.path.exists(reconstruct_path):
                    missing_data_count += 1
                    continue
                with open(reconstruct_path, 'r', encoding='utf-8') as f:
                    reconstructed_code = json.load(f).get("content", "")

                # 4. Construct Prompt
                direct_update_user_prompt = up.HIRE_DIRECT_UPDATE_EXPLANATION.format(
                    PROBLEM=problem_prompt,
                    ORIGINAL_CODE=original_code,
                    EXPLANATION=explanation,
                    RECONSTRUCTED_CODE=reconstructed_code
                )

                # 5. Create Batch Request
                model_name = resolve_model(update_model)
                is_together = is_together_model(model_name)
                
                request_body = {
                    "custom_id": task_id,
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": UPDATE_SYS},
                            {"role": "user", "content": direct_update_user_prompt}
                        ],
                    }
                }
                if not is_together:
                    request_body["method"] = "POST"
                    request_body["url"] = "/v1/chat/completions"
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Batch file generated: {output_file}")
    print(f"Requests created: {requests_created}")
    print(f"Skipped: {skipped_count}")
    print(f"Missing data: {missing_data_count}")
    
    if requests_created == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
        
    return output_file


def generate_self_refine_batch(args, dataset, cache_dir):
    """Generate batch requests for self-refining explanations by reflecting on original code."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating SELF_REFINE batch for items {start} to {end}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    self_refine_model = args.self_refine_model or args.eval_model
    l_suffix = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
    model_name_part = self_refine_model
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source or 'code'}_{args.eval_prompt}{l_suffix}_self_refine_{model_name_part}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_data_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks and str(task_id) not in args.tasks:
                    skipped_count += 1
                    continue

                # 1. Load original code
                if args.eval_source:
                    model_or_source = args.eval_source
                    raw_code = row.get(args.eval_source, "")
                else:
                    model_or_source = args.code_gen_model
                    gen_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.code_gen_model), f"{_sanitize_filename(task_id)}.json")
                    if os.path.exists(gen_cache_path):
                        with open(gen_cache_path, 'r', encoding='utf-8') as f:
                            raw_code = json.load(f).get("content", "")
                    else:
                        missing_data_count += 1
                        continue
                original_code = clean_code(raw_code)

                # 2. Load initial explanation
                is_query_aware = "query_aware" in (args.eval_prompt or "")
                is_objective = "explainer_obj" in (args.eval_prompt or "")
                source_prompt = "hire_explainer"
                if is_objective: source_prompt += "_obj"
                if is_query_aware: source_prompt += "_query_aware"
                source_folder = source_prompt
                if args.lambda_val is not None:
                     source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                
                source_model = args.explainer_model or args.eval_model
                explainer_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
                if not os.path.exists(explainer_path):
                    missing_data_count += 1
                    continue
                with open(explainer_path, 'r', encoding='utf-8') as f:
                    explanation = json.load(f).get("content", "")

                # 3. Construct Prompt
                self_refine_user_prompt = up.HIRE_SELF_REFINE_EXPLANATION.format(
                    PROBLEM=problem_prompt,
                    ORIGINAL_CODE=original_code,
                    EXPLANATION=explanation
                )

                # 4. Create Batch Request
                model_name = resolve_model(self_refine_model)
                is_together = is_together_model(model_name)
                
                request_body = {
                    "custom_id": task_id,
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": UPDATE_SYS},
                            {"role": "user", "content": self_refine_user_prompt}
                        ],
                    }
                }
                if not is_together:
                    request_body["method"] = "POST"
                    request_body["url"] = "/v1/chat/completions"
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Requests created: {requests_created}")
    print(f"Skipped: {skipped_count}")
    print(f"Missing data: {missing_data_count}")

    if requests_created == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
        
    print(f"Batch file generated: {output_file}")
    return output_file


def generate_style_transfer_batch(args, dataset, cache_dir):
    """Generate batch requests for explanation style transfer alignment."""
    start = args.start_problem
    end = min(args.end_problem if args.end_problem is not None else len(dataset), len(dataset))
    
    print(f"Generating STYLE_TRANSFER batch for items {start} to {end}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    explainer_model = args.explainer_model or args.eval_model
    style_transfer_model = args.style_transfer_model or explainer_model
    l_suffix = f"_L{args.lambda_val}" if args.lambda_val is not None else ""
    model_name_part = f"{explainer_model}_{style_transfer_model}" if style_transfer_model != explainer_model else explainer_model
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source or 'code'}_{args.eval_prompt}{l_suffix}_style_transfer_{model_name_part}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_data_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                if args.tasks and str(task_id) not in args.tasks:
                    skipped_count += 1
                    continue

                if args.eval_source:
                    model_or_source = args.eval_source
                else:
                    model_or_source = args.code_gen_model

                source_prompt = args.eval_prompt.replace("_query_aware", "")
                source_folder = source_prompt
                if args.lambda_val is not None:
                     source_folder = f"{source_prompt}_lambda_L{args.lambda_val}"
                
                source_model = args.explainer_model or args.eval_model
                explainer_path = os.path.join(cache_dir, args.dataset, model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
                
                if not os.path.exists(explainer_path):
                    missing_data_count += 1
                    continue
                with open(explainer_path, 'r', encoding='utf-8') as f:
                    explanation = json.load(f).get("content", "")

                style_transfer_user_prompt = up.HIRE_EXPLAINER_STYLE_TRANSFER.format(
                    PROBLEM=problem_prompt,
                    EXPLANATION=explanation
                )

                model_name = resolve_model(style_transfer_model)
                is_together = is_together_model(model_name)
                
                request_body = {
                    "custom_id": task_id,
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": "You are an expert technical editor. Your goal is to rewrite a code explanation to semantically align it with a problem statement. Crucially, the explanation is for a potentially buggy/incorrect code implementation: you MUST strictly preserve all logical bugs, incorrect execution details, and incorrect walkthrough outputs described in the original explanation."},
                            {"role": "user", "content": style_transfer_user_prompt}
                        ],
                    }
                }
                if not is_together:
                    request_body["method"] = "POST"
                    request_body["url"] = "/v1/chat/completions"
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Requests created: {requests_created}")
    print(f"Skipped: {skipped_count}")
    print(f"Missing data: {missing_data_count}")
    
    if requests_created == 0:
        if os.path.exists(output_file):
            os.remove(output_file)
        return None
        
    print(f"Batch file generated: {output_file}")
    return output_file


def generate_refine_batch(args, dataset, cache_dir):
    """Generate batch requests for code refinement based on evaluations."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    assert args.eval_source, "eval_source must be provided for refinement"
    
    refine_model = args.refine_model or args.code_gen_model
    print(f"Generating REFINEMENT batch for items {start} to {end} using model {refine_model}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filename includes eval_source and k and lambda if needed
    k_suffix = f"_k{args.k}" if (args.eval_prompt == "vanilla" and args.k > 1) else ""
    n_suffix = f"_N_{args.n}" if (args.eval_prompt.startswith("hire_") and "explainer" not in args.eval_prompt and "pseudo" not in args.eval_prompt) else ""
    l_suffix = f"_L{args.lambda_val}" if (args.lambda_val is not None and f"_L{args.lambda_val}" not in args.eval_prompt) else ""
    model_name_part = f"{args.explainer_model}_{args.eval_model}" if args.explainer_model else args.eval_model
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source}_{args.eval_prompt}{n_suffix}{k_suffix}{l_suffix}_refine_{model_name_part}_{refine_model}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_code_count = 0
    missing_eval_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
                if "incorrect_solution_old" in row or "wrong_canonical" in row:
                    skipped_count += 1
                    continue
                
                # 1. Load initial code from dataset row
                initial_code = row.get(args.eval_source, "")
                if not initial_code:
                    missing_code_count += 1
                    continue
                
                # 2. Load evaluation from cache
                evaluation_text = ""
                
                if args.eval_prompt == "vanilla" and args.k > 1:
                    # Concatenate multiple vanilla evaluations
                    reasonings = []
                    for k_idx in range(1, args.k + 1):
                        eval_prompt_type = f"vanilla_k{k_idx}"
                        model_folder = f"{args.explainer_model}_{args.eval_model}" if args.explainer_model else args.eval_model
                        eval_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.eval_source), _sanitize_filename(model_folder), _sanitize_filename(eval_prompt_type), f"{_sanitize_filename(task_id)}.json")
                        
                        if os.path.exists(eval_cache_path):
                            with open(eval_cache_path, 'r', encoding='utf-8') as f:
                                eval_data = json.load(f)
                                content = eval_data.get("content", "")
                                # Extract reasoning from JSON if possible
                                if '"reasoning":' in content:
                                    try:
                                        r_start = content.find('"reasoning":') + len('"reasoning":')
                                        r_end = content.rfind('}')
                                        reasoning = content[r_start:r_end].strip().strip('"').replace('\\n', '\n').replace('\\"', '"')
                                        reasonings.append(f"Evaluation {k_idx}: {reasoning}")
                                    except:
                                        reasonings.append(f"Evaluation {k_idx}: {content}")
                                else:
                                    reasonings.append(f"Evaluation {k_idx}: {content}")
                    
                    if not reasonings:
                        missing_eval_count += 1
                        continue
                    evaluation_text = "\n\n".join(reasonings)
                else:
                    # Single evaluation
                    eval_prompt_type = args.eval_prompt
                    if args.lambda_val is not None:
                        suffix = f"_L{args.lambda_val}"
                        if "_lambda" not in eval_prompt_type:
                            eval_prompt_type = f"{args.eval_prompt}_lambda{suffix}"
                        elif suffix not in eval_prompt_type:
                            eval_prompt_type = f"{args.eval_prompt}{suffix}"
                    elif (args.eval_prompt.startswith("hire_") and 
                          "explainer" not in args.eval_prompt and 
                          "pseudo" not in args.eval_prompt):
                        eval_prompt_type += f"_N_{args.n}"
                        
                    model_folder = f"{args.explainer_model}_{args.eval_model}" if args.explainer_model else args.eval_model
                    eval_cache_path = os.path.join(cache_dir, args.dataset, _sanitize_filename(args.eval_source), _sanitize_filename(model_folder), _sanitize_filename(eval_prompt_type), f"{_sanitize_filename(task_id)}.json")
                    
                    if not os.path.exists(eval_cache_path):
                        missing_eval_count += 1
                        continue
                    
                    with open(eval_cache_path, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)
                        evaluation_text = eval_data.get("content", "")
                    
                    # Standardize extraction of reasoning
                    if '"reasoning":' in evaluation_text:
                        try:
                            r_start = evaluation_text.find('"reasoning":') + len('"reasoning":')
                            r_end = evaluation_text.rfind('}')
                            evaluation_text = evaluation_text[r_start:r_end].strip().strip('"').replace('\\n', '\n').replace('\\"', '"')
                        except:
                            pass

                # 3. Construct Refinement Prompt
                refine_user_prompt = up.REFINE_PROMPT.format(
                    PROBLEM=problem_prompt,
                    CODE=initial_code,
                    EVALUATION=evaluation_text
                )

                # Determine language for system prompt
                if args.dataset in set(["leetcode", "humaneval_py", "debugbench_hard"]):
                    programming_language = "python"
                elif args.dataset == "humaneval_js":
                    programming_language = "javascript"
                elif args.dataset == "humaneval_java":
                    programming_language = "java"
                elif args.dataset == "humaneval_cpp":
                    programming_language = "c++"
                elif args.dataset == "humaneval_go":
                    programming_language = "go"
                else:
                    programming_language = "python" # fallback

                # 4. Create Batch Request
                refine_model_name = resolve_model(refine_model)
                is_together = is_together_model(refine_model_name)

                if is_together:
                    request_body = {
                        "custom_id": task_id,
                        "body": {
                            "model": refine_model_name,
                            "messages": [
                                {"role": "system", "content": up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=programming_language.upper(), 
                                                                PROGRAM_LANGUAGE_LOWER=programming_language)},
                                {"role": "user", "content": refine_user_prompt}
                            ],
                        }
                    }
                else:
                    request_body = {
                        "custom_id": task_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": refine_model_name,
                            "messages": [
                                {"role": "system", "content": up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=programming_language.upper(), 
                                                                PROGRAM_LANGUAGE_LOWER=programming_language)},
                                {"role": "user", "content": refine_user_prompt}
                            ],
                        }
                    }
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Batch file generated: {output_file}")
    print(f"Requests created: {requests_created}")
    print(f"Skipped (missing code): {missing_code_count}")
    print(f"Skipped (missing eval): {missing_eval_count}")
    
    return output_file

def submit_batch(batch_file):
    """Submit a batch file to OpenAI or Together AI API."""
    if not os.path.exists(batch_file):
        print(f"Error: Batch file {batch_file} does not exist.")
        sys.exit(1)
    
    # Peek at first line to determine if it's Together or OpenAI
    is_together = False
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            first_line = json.loads(f.readline())
            if "method" not in first_line:
                is_together = True
    except Exception as e:
        print(f"Error reading batch file for detection: {e}")
        sys.exit(1)

    if is_together:
        print(f"Detected Together AI batch file. Submitting...")
        try:
            client = Together()
            file_resp = client.files.upload(file=batch_file, purpose="batch-api", check=False)
            file_id = file_resp.id
            print(f"File uploaded. ID: {file_id}")

            batch_job = client.batches.create(input_file_id=file_id, endpoint="/v1/chat/completions")
            # BatchCreateResponse contains the BatchJob in the 'job' attribute
            job = batch_job.job
            job_id = job.id
            print(f"Together Batch job created. ID: {job_id}")
            
            # Save response
            output_dir = "batch_job_metadata"
            os.makedirs(output_dir, exist_ok=True)
            input_basename = os.path.basename(batch_file).replace('.jsonl', '')
            output_file = os.path.join(output_dir, f"{input_basename}_job.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Together job object might be a simple object or pydantic-like
                if hasattr(job, 'model_dump'):
                    f.write(json.dumps(job.model_dump(), indent=2, default=str))
                elif hasattr(job, 'dict'):
                    f.write(json.dumps(job.dict(), indent=2, default=str))
                else:
                    f.write(json.dumps(job.__dict__, indent=2, default=str))
            
            print(f"Job details saved to: {output_file}")
            print(f"\nTo check status later, run: python download_batch.py --job_file {output_file}")

        except Exception as e:
            print(f"Error submitting Together batch: {e}")
            sys.exit(1)
    else:
        print(f"Detected OpenAI batch file. Submitting...")
        try:
            client = OpenAI()
            batch_input_file = client.files.create(
                file=open(batch_file, "rb"),
                purpose="batch"
            )
            file_id = batch_input_file.id
            print(f"File uploaded. ID: {file_id}")
            
            print("Creating batch job...")
            batch_job = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": f"Batch job for {os.path.basename(batch_file)}"
                }
            )
            
            job_id = batch_job.id
            print(f"OpenAI Batch job created. ID: {job_id}")
            
            # Save response
            output_dir = "batch_job_metadata"
            os.makedirs(output_dir, exist_ok=True)
            
            input_basename = os.path.basename(batch_file).replace('.jsonl', '')
            output_file = os.path.join(output_dir, f"{input_basename}_job.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                if hasattr(batch_job, 'model_dump_json'):
                    f.write(batch_job.model_dump_json(indent=2))
                elif hasattr(batch_job, 'to_json'):
                    f.write(json.dumps(batch_job.to_json(), indent=2))
                else:
                    try:
                        f.write(json.dumps(batch_job.__dict__, indent=2, default=str))
                    except:
                        f.write(str(batch_job))
            
            print(f"Job details saved to: {output_file}")
            print(f"\nTo check status later, run: python download_batch.py --job_file {output_file}")
            
        except Exception as e:
            print(f"Error submitting OpenAI batch: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API requests for code generation or evaluation.")
    parser.add_argument("--mode", type=str, required=True, choices=["code", "eval", "refine", "reconstruct", "compare", "update", "direct_update", "self_refine", "dialogue", "dialogue_n_questions", "dialogue_n_questions_rc", "dialogue_n_questions_rc_exact", "style_transfer"], 
                        help="Mode: 'code' for code generation, 'eval' for evaluation, 'refine' for code refinement, 'reconstruct' for code reproduction from explanations, 'compare' for explanation feedback, 'update' for explanation improvement, 'direct_update' for direct explanation improvement skipping feedback, 'self_refine' for explanation refinement by self-reflection, 'dialogue' for multi-turn judge-explainer dialogue, 'dialogue_n_questions' for N-question dialogue evaluation, 'style_transfer' for explanation semantic alignment")
    parser.add_argument("--dataset", type=str, required=True, nargs='+', help="Name of the dataset(s) (e.g., 'leetcode')")
    parser.add_argument("--code_gen_model", type=str, required=False, help="Model for code generation (e.g., 'gpt-4o-mini')")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save batch files (default: batch_jobs for code, batch_jobs_eval for eval)")
    
    # Evaluation-specific arguments
    parser.add_argument("--eval_model", type=str, nargs='+', help="Model(s) for evaluation (required if mode=eval)", default=["gpt-4o-mini"])
    parser.add_argument("--eval_prompt", type=str, nargs='+', 
                        help="Evaluation prompt type(s) (required if mode=eval)")
    parser.add_argument("--analysis_model", type=str, help="Model used for analysis (only for cj_summary, defaults to eval_model)")
    parser.add_argument("--explainer_model", type=str, help="Source model for dependencies like explanations/pseudocode (defaults to eval_model)")
    parser.add_argument("--n", type=int, default=3, help="Number of steps for hire_decomposer")
    parser.add_argument("--k", type=int, default=1, help="Number of independent evaluations for vanilla prompt")
    parser.add_argument("--lambda_val", type=int, nargs='+', default=None, help="Controllable semantic abstraction parameter(s) (1, 3, 5, 8, 10)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for evaluation (default: 0.0, set to 1.0 if k > 1)")
    parser.add_argument("--refine_model", type=str, help="Model for code refinement (defaults to code_gen_model if mode=refine)")
    parser.add_argument("--reconstruct_model", type=str, help="Model for code reconstruction (defaults to eval_model)")
    parser.add_argument("--compare_model", type=str, help="Model for explanation feedback (defaults to eval_model)")
    parser.add_argument("--update_model", type=str, help="Model for explanation improvement (defaults to eval_model)")
    parser.add_argument("--self_refine_model", type=str, help="Model for explanation self-refinement (defaults to eval_model)")
    parser.add_argument("--style_transfer_model", type=str, help="Model for explanation style-transfer (defaults to explainer_model)")
    parser.add_argument("--compact_second_judge", action="store_true", help="Use HIRE_DIALOGUE_SECOND_JUDGE_USER_RC_COMPACT for failed rc_exact judge decisions")
    parser.add_argument("-requirements_only_judges", "--requirements_only_judges", dest="requirements_only_judges", action="store_true", help="Extract requirements first and expose judges only to that requirements list")
    parser.add_argument("--requirements_model", type=str, default=None, help="Model for requirements extraction (defaults to eval_model)")
    parser.add_argument("--force_requirements", action="store_true", help="Regenerate shared requirements even when they are already cached")
    parser.add_argument("-specifications_only_judges", "--specifications_only_judges", dest="specifications_only_judges", action="store_true", help="Extract specifications first and expose judges only to that specification list")
    parser.add_argument("--specifications_model", type=str, default=None, help="Model for specification extraction (defaults to eval_model)")
    parser.add_argument("--force_specifications", action="store_true", help="Regenerate shared specifications even when they are already cached")
    parser.add_argument("--one_by_one_verifier", action="store_true", help="Verify conservative specifications independently and clarify only uncertain requirements")
    parser.add_argument("--conservative_specifications_model", type=str, default=None, help="Model for conservative specification extraction (defaults to eval_model)")
    parser.add_argument("--force_conservative_specifications", action="store_true", help="Regenerate shared conservative specifications")
    
    # Submission control
    parser.add_argument("--dont_submit", action="store_true", help="Don't submit batch to OpenAI (only generate the file)")
    parser.add_argument("--eval_source", type=str, nargs='+', help="Key(s) in dataset to evaluate instead of generated code (e.g. 'canonical_solution')", default=["canonical_solution"])
    parser.add_argument("--force", action="store_true", help="Force generation even if item is already in cache")
    parser.add_argument("--tasks", type=str, nargs='+', help="Specific task ID(s) to process")
    parser.add_argument("--dry_run", action="store_true", help="Alias for --dont_submit")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for direct evaluation (llama31-8b)")
    parser.add_argument("--run_normal", action="store_true", help="Run evaluation directly using normal API instead of Batch API")


    args = parser.parse_args()
    if args.requirements_only_judges and args.mode != "dialogue_n_questions":
        parser.error("--requirements_only_judges should be used with --mode dialogue_n_questions; RC modes are redundant")
    if args.specifications_only_judges and args.mode != "dialogue_n_questions":
        parser.error("--specifications_only_judges should be used with --mode dialogue_n_questions; RC modes are redundant")
    if args.requirements_only_judges and args.specifications_only_judges:
        parser.error("--requirements_only_judges and --specifications_only_judges are mutually exclusive")
    if args.one_by_one_verifier and args.mode != "dialogue_n_questions":
        parser.error("--one_by_one_verifier should be used with --mode dialogue_n_questions")
    if args.one_by_one_verifier and (args.requirements_only_judges or args.specifications_only_judges):
        parser.error("--one_by_one_verifier cannot be combined with requirements-only or specifications-only judges")
    
    if args.mode in ["dialogue", "dialogue_n_questions", "dialogue_n_questions_rc", "dialogue_n_questions_rc_exact"] and not args.run_normal:
        parser.error(f"--mode {args.mode} requires --run_normal because interactive query-answering cannot run via offline batch API.")

    if args.dry_run:
        args.dont_submit = True
    
    # Set default output directory based on mode
    if args.output_dir is None:
        if args.mode == "code":
            args.output_dir = "batch_jobs"
        elif args.mode == "eval":
            args.output_dir = "batch_jobs_eval"
        elif args.mode == "reconstruct":
            args.output_dir = "batch_jobs_reconstruct"
        elif args.mode == "compare":
            args.output_dir = "batch_jobs_compare"
        elif args.mode == "update":
            args.output_dir = "batch_jobs_update"
        elif args.mode == "direct_update":
            args.output_dir = "batch_jobs_direct_update"
        elif args.mode == "self_refine":
            args.output_dir = "batch_jobs_self_refine"
        elif args.mode in ["dialogue", "dialogue_n_questions", "dialogue_n_questions_rc", "dialogue_n_questions_rc_exact"]:
            args.output_dir = "batch_jobs_dialogue"
        elif args.mode == "style_transfer":
            args.output_dir = "batch_jobs_style_transfer"
        else: # refine
            args.output_dir = "batch_jobs_refine"

    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    datasets = args.dataset
    eval_models = args.eval_model if args.eval_model else [None]
    eval_prompts = args.eval_prompt if args.eval_prompt else [None]
    eval_sources = args.eval_source if args.eval_source else [None]
    lambda_vals = args.lambda_val if args.lambda_val else [None]

    all_batch_files = []

    for d_name in datasets:
        print(f"\nProcessing dataset: {d_name}")
        # Load dataset
        try:
            dataset = CodeData(d_name)
        except Exception as e:
            print(f"Error loading dataset {d_name}: {e}")
            continue

        # Create a copy of args to modify for this combination
        current_args = copy.deepcopy(args)
        current_args.dataset = d_name

        if current_args.end_problem is None:
            current_args.end_problem = len(dataset)

        for e_model in eval_models:
            current_args.eval_model = e_model
            for e_prompt in eval_prompts:
                current_args.eval_prompt = e_prompt

                # Guardrail: lambda_val only matters for lambda-aware methods (e.g. hire_explainer/hire_pseudo families)
                effective_lambda_vals = lambda_vals
                is_lambda_aware = any(x in e_prompt for x in ["alignment", "explainer", "pseudo", "explanation"]) if e_prompt else False
                if not is_lambda_aware:
                    effective_lambda_vals = [None]
                
                for e_source in eval_sources:
                    current_args.eval_source = e_source
                    for l_val in effective_lambda_vals:
                        current_args.lambda_val = l_val
                        
                        # Validate mode-specific arguments for this combination
                        if current_args.mode == "eval":
                            if not current_args.eval_model or not current_args.eval_prompt:
                                print(f"Skipping combination: mode=eval requires eval_model and eval_prompt")
                                continue
                            if current_args.eval_prompt == "cj_summary" and not current_args.analysis_model:
                                current_args.analysis_model = current_args.eval_model
                        
                        if current_args.mode in ["dialogue", "dialogue_n_questions", "dialogue_n_questions_rc", "dialogue_n_questions_rc_exact"]:
                            if not current_args.eval_model or not current_args.eval_prompt:
                                print(f"Skipping combination: mode={current_args.mode} requires eval_model and eval_prompt")
                                continue
                            if not current_args.explainer_model:
                                current_args.explainer_model = current_args.eval_model
                                
                        if current_args.mode == "refine":
                            if not current_args.eval_model or not current_args.eval_prompt:
                                print(f"Skipping combination: mode=refine requires eval_model and eval_prompt")
                                continue
                            if not current_args.eval_source:
                                print(f"Skipping combination: mode=refine requires eval_source")
                                continue
                            if not current_args.refine_model:
                                current_args.refine_model = current_args.code_gen_model or "gpt-4o-mini"

                        # Generate batch based on mode
                        batch_files = []
                        if current_args.run_normal:
                             print(f"\n[Direct Execution] --run_normal flag detected for mode '{current_args.mode}'. Running directly instead of batch.")
                             run_direct_mode(current_args, dataset, cache_dir)
                             batch_file = None
                        elif current_args.mode == "code":
                            batch_file = generate_code_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "eval":
                             if current_args.k > 1:
                                temperature = current_args.temperature if current_args.temperature != 0.0 else 1.0
                                for k in range(1, current_args.k + 1):
                                    batch_file = generate_eval_batch(current_args, dataset, cache_dir, k_index=k, temperature=temperature)
                                    if batch_file: batch_files.append(batch_file)
                             else:
                                batch_file = generate_eval_batch(current_args, dataset, cache_dir, k_index=None, temperature=current_args.temperature)
                                if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "refine":
                            batch_file = generate_refine_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "reconstruct":
                            batch_file = generate_reconstruct_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "compare":
                            batch_file = generate_compare_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "update":
                            batch_file = generate_update_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "direct_update":
                            batch_file = generate_direct_update_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "self_refine":
                            batch_file = generate_self_refine_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        elif current_args.mode == "style_transfer":
                            batch_file = generate_style_transfer_batch(current_args, dataset, cache_dir)
                            if batch_file: batch_files.append(batch_file)
                        
                        all_batch_files.extend(batch_files)

    # Submit batches if not disabled
    if not args.dont_submit:
        if all_batch_files:
            print("\n" + "="*50)
            print(f"Submitting {len(all_batch_files)} batches to API...")
            print("="*50)
            for batch_file in all_batch_files:
                if batch_file:
                    submit_batch(batch_file)
        else:
            print("\nNo batches to submit.")
    else:
        print("\nBatch files created but not submitted (--dont_submit flag set)")
        for batch_file in all_batch_files:
            print(f"To submit later, run: python run_batch.py --batch_input_file {batch_file}")

if __name__ == "__main__":
    main()
