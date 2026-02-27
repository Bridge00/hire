import argparse
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

from data.all_code_benchmarks import CodeData
#from utils.prompts import CODEGEN_SYS, VANILLA_EVAL_BINARY, CODEJUDGE_ANALYSIS, CODEJUDGE_SUMMARY, HIRE_DECOMPOSER
import utils.prompts as up
from utils.llm import clean_code

load_dotenv()

# From TextGrad
EVAL_SYS = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise critiques."

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
                task_id, prompt, _, _ = dataset[i]
                
                # Check structured cache: dataset/code_gen_model/task_id.json
                cache_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, f"{task_id}.json")
                
                if not args.force and os.path.exists(cache_path):
                    skipped_count += 1
                    continue
                
                # Create Batch Request
                if args.dataset in set(["leetcode", "humaneval_py"]):
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

                request_body = {
                        "custom_id": task_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                        "model": args.code_gen_model,
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
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating EVALUATION batch for items {start} to {end} with prompt '{args.eval_prompt}'")
    
    if args.eval_source:
        code_source = args.eval_source
        model_or_source = args.eval_source
    else:
        code_source = f'code_model_{args.code_gen_model}'
        model_or_source = args.code_gen_model
        if not model_or_source:
             print("Error: --code_gen_model is required when --eval_source is not provided")
             return None

    eval_prompt_type = args.eval_prompt
    if args.eval_prompt.startswith("hire_") and args.eval_prompt not in ["hire_explainer", "hire_explainer_query_aware", "hire_explainer_checker", "hire_explainer_checker_query_aware"]:
        eval_prompt_type = f"{args.eval_prompt}_N_{args.n}"
        
    if k_index is not None:
        eval_prompt_type = f"{eval_prompt_type}_k{k_index}"

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, 
        f"{args.dataset}_{code_source}_{eval_prompt_type}_eval_eval_{args.eval_model}_{start}_{end}.jsonl"
    )
    
    requests_created = 0
    skipped_count = 0
    missing_code_count = 0
    missing_code_count = 0
    missing_analysis_count = 0
    missing_plan_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:


        for i in range(start, end):
            try:
                task_id, problem_prompt, _, canonical_solution, row = dataset[i]
                
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
                    gen_cache_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, f"{task_id}.json")
                    
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

                elif args.eval_prompt == "hire_explainer":
                    eval_user_prompt = up.HIRE_EXPLAINER.format(CODE=cleaned_code)
                    active_eval_prompt_type = "hire_explainer"
                elif args.eval_prompt == "hire_explainer_query_aware":
                    eval_user_prompt = up.HIRE_EXPLAINER_QUERY_AWARE.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    active_eval_prompt_type = "hire_explainer_query_aware"

                elif args.eval_prompt in ["hire_explainer_checker", "hire_explainer_checker_query_aware"]:
                    source_prompt = "hire_explainer" if args.eval_prompt == "hire_explainer_checker" else "hire_explainer_query_aware"
                    
                    # Check structured cache for the explanation
                    explainer_path = os.path.join(cache_dir, args.dataset, model_or_source, args.eval_model, source_prompt, f"{task_id}.json")
                    
                    if not os.path.exists(explainer_path):
                        missing_eval_count += 1
                        continue
                        
                    with open(explainer_path, 'r', encoding='utf-8') as f:
                        expl_data = json.load(f)
                        explanation = expl_data.get("content", "")
                        
                    eval_user_prompt = up.HIRE_EXPLAINER_CHECKER.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
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
                        
                        decomposed_plan_path = os.path.join(cache_dir, args.dataset, model_or_source, args.eval_model, source_decomposer_base, f"{task_id}.json")
                        
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
                             plan_path = os.path.join(cache_dir, args.dataset, model_or_source, args.eval_model, plan_folder, f"{task_id}.json")
                             
                             # Dependency 2: Commentor
                             commentor_suffix = "_query_aware_flexible" if is_query_aware else "_flexible"
                             commentor_folder = f"hire_commentor_checker{commentor_suffix}_N_{args.n}"
                             commentor_path = os.path.join(cache_dir, args.dataset, model_or_source, args.eval_model, commentor_folder, f"{task_id}.json")
                             
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
                                step_cache_path = os.path.join(cache_dir, args.dataset, model_or_source, args.eval_model, step_folder, f"{task_id}.json")
                                if not args.force and os.path.exists(step_cache_path):
                                    skipped_count += 1
                                    continue

                                request_body = {
                                    "custom_id": step_task_id,
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": args.eval_model,
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
                    analysis_model = args.analysis_model or args.eval_model
                    analysis_path = os.path.join(cache_dir, args.dataset, model_or_source, analysis_model, "cj_analysis", f"{task_id}.json")
                    
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
                        eval_cache_path = os.path.join(cache_dir, args.dataset, args.eval_source, args.eval_model, cache_prompt_type, f"{task_id}.json")
                else:
                        eval_cache_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, args.eval_model, cache_prompt_type, f"{task_id}.json")

                if not args.force and os.path.exists(eval_cache_path):
                    skipped_count += 1
                    continue

                request_body = {
                    "custom_id": task_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.eval_model,
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
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Batch file generated: {output_file}")
    print(f"Total items in range: {end - start}")
    print(f"Requests created: {requests_created}")
    print(f"Skipped (already cached): {skipped_count}")
    print(f"Skipped (missing generated code): {missing_code_count}")
    if args.eval_prompt == "cj_summary":
        print(f"Skipped (missing analysis): {missing_analysis_count}")
    if args.eval_prompt in ["hire_plan_checker", "hire_implementation_checker_isolated", "hire_implementation_checker_context"]:
         print(f"Skipped (missing decomposed plan): {missing_plan_count}")

    if requests_created == 0:
        print(f"No requests created. Deleting empty file: {output_file}")
        try:
            os.remove(output_file)
        except OSError:
            pass
        return None
    
    return output_file

def generate_refine_batch(args, dataset, cache_dir):
    """Generate batch requests for code refinement based on evaluations."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    assert args.eval_source, "eval_source must be provided for refinement"
    
    refine_model = args.refine_model or args.code_gen_model
    print(f"Generating REFINEMENT batch for items {start} to {end} using model {refine_model}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filename includes eval_source and k if needed
    k_suffix = f"_k{args.k}" if (args.eval_prompt == "vanilla" and args.k > 1) else ""
    n_suffix = f"_N_{args.n}" if (args.eval_prompt.startswith("hire_") and args.eval_prompt not in ["hire_explainer", "hire_explainer_query_aware", "hire_explainer_checker", "hire_explainer_checker_query_aware"]) else ""
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.eval_source}_{args.eval_prompt}{n_suffix}{k_suffix}_eval_{args.eval_model}_{refine_model}_refine_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0
    missing_code_count = 0
    missing_eval_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _, row = dataset[i]
                
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
                        eval_cache_path = os.path.join(cache_dir, args.dataset, args.eval_source, args.eval_model, eval_prompt_type, f"{task_id}.json")
                        
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
                    if args.eval_prompt.startswith("hire_") and args.eval_prompt not in ["hire_explainer", "hire_explainer_query_aware", "hire_explainer_checker", "hire_explainer_checker_query_aware"]:
                        eval_prompt_type += f"_N_{args.n}"
                        
                    eval_cache_path = os.path.join(cache_dir, args.dataset, args.eval_source, args.eval_model, eval_prompt_type, f"{task_id}.json")
                    
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
                if args.dataset in set(["leetcode", "humaneval_py"]):
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
                request_body = {
                    "custom_id": task_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": refine_model,
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
    """Submit a batch file to OpenAI API."""
    if not os.path.exists(batch_file):
        print(f"Error: Batch file {batch_file} does not exist.")
        sys.exit(1)
    
    try:
        client = OpenAI()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file or environment variables.")
        sys.exit(1)
    
    print(f"Uploading file: {batch_file}")
    
    try:
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
        print(f"Batch job created. ID: {job_id}")
        
        # Save response
        output_dir = "batch_job_metadata"
        os.makedirs(output_dir, exist_ok=True)
        
        input_basename = os.path.basename(batch_file)
        # Remove extension for cleaner name
        if input_basename.endswith('.jsonl'):
            input_basename = input_basename[:-6]
        
        output_file = os.path.join(output_dir, f"{input_basename}_job.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # batch_job is a Pydantic model in recent SDKs
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
        print(f"Error submitting batch: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API requests for code generation or evaluation.")
    parser.add_argument("--mode", type=str, required=True, choices=["code", "eval", "refine"], 
                        help="Mode: 'code' for code generation, 'eval' for evaluation, 'refine' for code refinement")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'leetcode')")
    parser.add_argument("--code_gen_model", type=str, required=False, help="Model for code generation (e.g., 'gpt-4o-mini')")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save batch files (default: batch_jobs for code, batch_jobs_eval for eval)")
    
    # Evaluation-specific arguments
    parser.add_argument("--eval_model", type=str, help="Model for evaluation (required if mode=eval)", default="gpt-4o-mini")
    parser.add_argument("--eval_prompt", type=str, choices=[
        "vanilla", "cj_analysis", "cj_summary", "cj_fault_localization", 
        "hire_decomposer", "hire_plan_checker", "hire_implementation_checker_isolated", "hire_implementation_checker_context",
        "hire_decomposer_query_aware", "hire_plan_checker_query_aware", "hire_implementation_checker_isolated_query_aware", "hire_implementation_checker_context_query_aware",
        "hire_decomposer_flexible", "hire_decomposer_query_aware_flexible",
        "hire_plan_checker_flexible", "hire_plan_checker_query_aware_flexible",
        "hire_implementation_checker_isolated_flexible", "hire_implementation_checker_context_flexible",
        "hire_implementation_checker_isolated_query_aware_flexible", "hire_implementation_checker_context_query_aware_flexible",
        "hire_plan_checker_text_only_flexible", "hire_plan_checker_query_aware_text_only_flexible",
        "hire_commentor_checker_flexible", "hire_commentor_checker_query_aware_flexible",
        "hire_aggregator_flexible", "hire_aggregator_query_aware_flexible",
        "hire_aggregator_a2_aware_flexible", "hire_aggregator_a2_aware_query_aware_flexible",
        "hire_explainer", "hire_explainer_query_aware",
        "hire_explainer_checker", "hire_explainer_checker_query_aware",
        "ice_correctness", "ice_usefulness"], 
                        help="Evaluation prompt type (required if mode=eval)")
    parser.add_argument("--analysis_model", type=str, help="Model used for analysis (only for cj_summary, defaults to eval_model)")
    parser.add_argument("--n", type=int, default=3, help="Number of steps for hire_decomposer")
    parser.add_argument("--k", type=int, default=1, help="Number of independent evaluations for vanilla prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for evaluation (default: 0.0, set to 1.0 if k > 1)")
    parser.add_argument("--refine_model", type=str, help="Model for refinement (defaults to code_gen_model if mode=refine)")
    
    # Submission control
    parser.add_argument("--dont_submit", action="store_true", help="Don't submit batch to OpenAI (only generate the file)")
    parser.add_argument("--eval_source", type=str, help="Key in dataset to evaluate instead of generated code (e.g. 'canonical_solution')")
    parser.add_argument("--force", action="store_true", help="Force generation even if item is already in cache")


    args = parser.parse_args()
    
    # Set default output directory based on mode
    if args.output_dir is None:
        if args.mode == "code":
            args.output_dir = "batch_jobs"
        elif args.mode == "eval":
            args.output_dir = "batch_jobs_eval"
        else: # refine
            args.output_dir = "batch_jobs_refine"
    
    # Validate mode-specific arguments
    if args.mode == "eval":
        if not args.eval_model or not args.eval_prompt:
            print("Error: --eval_model and --eval_prompt are required when mode=eval")
            sys.exit(1)
        if args.eval_prompt == "cj_summary" and not args.analysis_model:
            args.analysis_model = args.eval_model
    
    if args.mode == "refine":
        if not args.eval_model or not args.eval_prompt:
            print("Error: --eval_model and --eval_prompt are required when mode=refine to locate evaluations in cache")
            sys.exit(1)
        if not args.eval_source:
            print("Error: --eval_source is REQUIRED when mode=refine")
            sys.exit(1)
        if not args.refine_model:
            args.refine_model = args.code_gen_model or "gpt-4o-mini"

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = CodeData(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.end_problem is None:
        args.end_problem = len(dataset)
    
    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    
    # Generate batch based on mode
    batch_files = []
    if args.mode == "code":
        batch_file = generate_code_batch(args, dataset, cache_dir)
        batch_files.append(batch_file)
    elif args.mode == "eval":
        if args.k > 1:
            # Default temperature to 1.0 if k > 1 and not explicitly set by user to something else non-zero
            temperature = args.temperature if args.temperature != 0.0 else 1.0
            for k in range(1, args.k + 1):
                batch_file = generate_eval_batch(args, dataset, cache_dir, k_index=k, temperature=temperature)
                batch_files.append(batch_file)
        else:
            batch_file = generate_eval_batch(args, dataset, cache_dir, k_index=None, temperature=args.temperature)
            batch_files.append(batch_file)
    elif args.mode == "refine":
        batch_file = generate_refine_batch(args, dataset, cache_dir)
        batch_files.append(batch_file)
    else:
        pass
    
    # Submit batches if not disabled
    if not args.dont_submit:
        print("\n" + "="*50)
        print("Submitting batches to OpenAI...")
        print("="*50)
        for batch_file in batch_files:
            if batch_file:
                submit_batch(batch_file)
    else:
        print("\nBatch files created but not submitted (--dont_submit flag set)")
        for batch_file in batch_files:
            print(f"To submit later, run: python run_batch.py --batch_input_file {batch_file}")

if __name__ == "__main__":
    main()
