from scipy.stats import kendalltau, spearmanr
from typing import List, Dict, Any
import re
import numpy as np
import pandas as pd
import json
import os
from data.all_code_benchmarks import CodeData
from utils.llm import _get_cache_key, _sanitize_filename
from utils.prompts import CODEJUDGE_ANALYSIS, CODEJUDGE_SUMMARY, VANILLA_EVAL_BINARY
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Counts the number of tokens in a text using tiktoken."""
    try:
        # GPT-5 and GPT-4o family use o200k_base
        if "gpt-5" in model or "gpt-4o" in model:
            encoding = tiktoken.get_encoding("o200k_base")
        else:
            encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base") # Fallback for unknown models (o200k is used by gpt-4o family)
    return len(encoding.encode(text or ""))

def rank_metrics(scores_a, scores_b) -> Dict[str, float]:
    kt = kendalltau(scores_a, scores_b, nan_policy="omit")
    sp = spearmanr(scores_a, scores_b, nan_policy="omit")
    return {
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
        "spearman_r": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
    }

def parse_score(response: str) -> float:
    """
    Parses a Yes/No response into a float score (1.0/0.0).
    Also tries to handle simple numbers if present.
    """
    if not response:
        return 0.0
    
    clean_response = response.strip().lower()
    
    # Check for Yes/No (handling punctuation)
    if "yes" in clean_response:
        return 1.0
    if "no" in clean_response:
        return 0.0
        
    # Try finding a number
    try:
        match = re.search(r"(\d+(\.\d+)?)", clean_response)
        if match:
            val = float(match.group(1))
            # Normalize if it looks like 0-100 score? 
            # For now assume mostly binary or 0-1 if number.
            if val > 1.0 and val <= 100.0:
                return val / 100.0
            return val
    except:
        pass
        
    return 0.0

def calculate_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates correlation metrics between LLM evaluation scores and Execution pass rates.
    
    Args:
        results: The dictionary output from run.py (containing "results" list).
    
    Returns:
        Dictionary of correlation metrics.
    """
    items = results.get("results", [])
    if not items:
        # Fallback if passed a list directly
        if isinstance(results, list):
            items = results
        else:
            return {}

    llm_scores = []
    exec_scores = []
    
    for item in items:
        # Extract Execution Score (Pass Rate)
        exec_metrics = item.get("execution_metrics", {})
        pass_rate = exec_metrics.get("pass_rate", 0.0)
        
        # Extract LLM Score
        llm_eval = item.get("LLM_evaluation", {})
        # Depending on evaluator, it might be in "response" or "score"
        if "score" in llm_eval:
            llm_score = float(llm_eval["score"])
        elif "response" in llm_eval:
            llm_score = parse_score(llm_eval["response"])
        else:
            llm_score = 0.0
            
        llm_scores.append(llm_score)
        exec_scores.append(int(pass_rate == 1))
        
    return rank_metrics(llm_scores, exec_scores)


def create_results_dataframe(dataset: str, 
                              code_gen_model: str, 
                              eval_model: str = None, 
                              evaluation_method: str = None, 
                              lambda_val: int = None,
                              seed: int = 95, N = 3,
                              checker_model: str = None,
                              explainer_model: str = None) -> pd.DataFrame:
    """
    Creates a pandas DataFrame combining execution results and LLM evaluations.
    
    Args:
        dataset: Name of the dataset (e.g., 'leetcode', 'humaneval')
        code_gen_model: Model used to generate code (e.g., 'gpt-4o-mini')
        eval_model: Legacy model name used for LLM evaluation (backward compatibility)
        evaluation_method: Evaluation method ('vanilla', 'codejudge', or None for execution-only)
        lambda_val: Lambda value for controllable strictness (optional)
        seed: Random seed used in the experiment
        checker_model: Model used for the checking stage
        explainer_model: Source model for dependencies like explanations/pseudocode (defaults to eval_model)
        
    Returns:
        DataFrame with columns: task_id, prompt, generated_code, state, feedback, pass_rate,
        and LLM evaluation columns (vanilla_eval or cj_analysis + cj_summary for CodeJudge)
    """
    # Define override keys
    PASS_RATE_1_KEYS = {"canonical_solution", "reverse_authority", "misleading_task_canonical"}
    PASS_RATE_0_KEYS = {"misleading_task_incorrect",  "authority", "incorrect_solution"}
    
    # Load execution results or fallback to dataset
    exec_log_path = os.path.join(os.getenv("ROOT_DIR"), "execution_logs", f"seed_{seed}_{dataset}_{code_gen_model}_exec.json")
    
    exec_data_results = []
    
    if os.path.exists(exec_log_path):
        with open(exec_log_path, 'r', encoding='utf-8') as f:
            exec_data = json.load(f)
            exec_data_results = exec_data.get('results', [])
    elif code_gen_model in PASS_RATE_1_KEYS or code_gen_model in PASS_RATE_0_KEYS:
        # Fallback: Load dataset directly if log is missing for override keys
        try:
            data_obj = CodeData(dataset)
            # data_obj is iterable/indexable. We need to reconstruct similar dict structure.
            # CodeData[i] -> (task_id, prompt, ..., ...)
            
            # Note: The usage below iterates exec_data['results'].
            # We'll construct a mock list.
            for i in range(len(data_obj)):
                 # CodeData returns: task_id, prompt, test_cases, canonical_solution, raw_row
                 tid, prmpt, _, canon, _ = data_obj[i]
                 
                 # For generated_code, we ideally want what was used. 
                 # If we don't have it, we put canonical or empty string.
                 # User said "we don't need to look into execution logs", so code might not strictly matter for metrics unless we show it.
                 # We'll use canonical so it's not empty.
                 
                 exec_data_results.append({
                     'task_id': tid,
                     'prompt': prmpt,
                     'generated_code': canon, # Placeholder
                     'state': [],
                     'feedback': '',
                     'pass_rate': 0.0 # Will be overridden
                 })
                 
        except Exception as e:
            print(f"Warning: Could not load dataset for fallback: {e}")
            raise FileNotFoundError(f"Execution log not found and fallback failed: {exec_log_path}")
    else:
        # Not a special key, so log MUST exist
        raise FileNotFoundError(f"Execution log not found: {exec_log_path}")

    
    # Build rows
    rows = []
    cache_dir = os.getenv("CACHE_DIR", ".cache")
    
    # Determine explainer and checker models
    explainer_to_use = explainer_model or eval_model
    checker_to_use = checker_model or eval_model
    
    if explainer_model and checker_model:
        model_folder = f"{explainer_model}_{checker_model}"
    elif checker_model and eval_model:
        # Backward compatibility for (eval_model='gpt-4o-mini', checker_model='qwen')
        model_folder = f"{eval_model}_{checker_model}"
    else:
        model_folder = checker_to_use
    
    for result in exec_data_results:
        task_id = result['task_id']
        sanitized_task_id = _sanitize_filename(task_id)
        prompt = result['prompt']
        generated_code = result['generated_code']
        state = tuple(result.get('state', []))
        feedback = result.get('feedback', '')
        
        # Apply Overrides
        if code_gen_model in PASS_RATE_1_KEYS:
            pass_rate = 1.0
        elif code_gen_model in PASS_RATE_0_KEYS:
            pass_rate = 0.0
        else:
            pass_rate = result.get('pass_rate', 0.0)
        
        row = {
            'task_id': task_id,
            'prompt': prompt,
            'generated_code': generated_code,
            'state': state,
            'feedback': feedback,
            'pass_rate': pass_rate
        }
        
        # Retrieve LLM evaluation from cache if evaluation_method is specified
        if evaluation_method and eval_model:
            vanilla_k_match = re.match(r"vanilla_(\d+)", evaluation_method)
            is_explanation_variant = "behavior_comparison_explanation" in evaluation_method or "two_phase_reflective_explanation" in evaluation_method
            
            if evaluation_method in ['vanilla', 'vanilla_no_reasoning', 'behavior_comparison', 'two_phase_reflective'] or is_explanation_variant:
                # Construct folder name
                folder_name = evaluation_method
                if lambda_val is not None and "explanation" in evaluation_method:
                    if "_lambda" not in folder_name:
                        folder_name = f"{folder_name}_lambda"
                    folder_name = f"{folder_name}_L{lambda_val}"

                # Get evaluation from structured cache
                # Path: dataset/code_gen_model/model_folder/prompt/task_id.json
                cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, folder_name, f"{sanitized_task_id}.json")
                
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        content = cached.get('content', '')
                        row[folder_name] = content
                        
                        content_lower = content.lower()
                        
                        # Handle JSON-based verdict for all these baselines
                        v_match = re.search(r'"correct":\s*(true|false|"yes"|"no"|1|0)', content_lower)
                        if v_match:
                            val = v_match.group(1).strip('"')
                            verdict = "Yes" if val in ['true', 'yes', '1'] else "No"
                            row[evaluation_method] = verdict # Backward compatibility for vanilla
                            if evaluation_method != 'vanilla':
                                row[f"{evaluation_method}_verdict"] = verdict
                        else:
                            # Fallback for plain text if JSON-like structure is missing
                            if "correct: yes" in content_lower or "verdict: yes" in content_lower:
                                verdict = "Yes"
                            elif "correct: no" in content_lower or "verdict: no" in content_lower:
                                verdict = "No"
                            else:
                                verdict = None
                            
                            if verdict:
                                if evaluation_method == 'vanilla':
                                    row["vanilla"] = verdict
                                else:
                                    row[f"{evaluation_method}_verdict"] = verdict
                else:
                    row[folder_name] = None
            elif vanilla_k_match:
                k = int(vanilla_k_match.group(1))
                scores = []
                for j in range(1, k + 1):
                    # Cache prompt type used in generate_batch.py was f"vanilla_k{j}"
                    cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, f"vanilla_k{j}", f"{sanitized_task_id}.json")
                    if os.path.exists(cache_path):
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached = json.load(f)
                            scores.append(parse_score(cached.get('content', '')))
                
                if scores:
                    # Majority vote: 1.0 if more than half are positive
                    vote_score = 1.0 if sum(scores) > len(scores) / 2 else 0.0
                    row[evaluation_method] = "Yes" if vote_score == 1.0 else "No"
                else:
                    row[evaluation_method] = None
                    
            elif evaluation_method == 'codejudge':
                # Get CodeJudge analysis
                # Path: dataset/code_gen_model/eval_model/cj_analysis/task_id.json
                
                for step in ['cj_analysis', 'cj_summary']:
                    cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, step, f"{sanitized_task_id}.json")
                    
                    if os.path.exists(cache_path):
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached = json.load(f)
                            content = cached.get('content', '')
                            if step == 'cj_summary' and content:
                                content = content.strip().rstrip('.')
                            row[step] = content
                    else:
                        row[step] = None
                

            elif "explainer" in evaluation_method:
                # hire_explainer, hire_explainer_query_aware, hire_explainer_checker, hire_explainer_checker_query_aware
                # These prompts do NOT use the _N_ suffix in the structured cache
                
                # Construct folder name considering lambda
                folder_name = evaluation_method
                if lambda_val is not None:
                    # Handle the _lambda suffix if it's missing from the method name
                    if "_lambda" not in folder_name:
                        folder_name = folder_name + "_lambda"
                    
                    folder_name = f"{folder_name}_L{lambda_val}"
                
                cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, folder_name, f"{sanitized_task_id}.json")
                
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        content = cached.get('content', '')
                        row[folder_name] = content
                        
                        if "checker" in evaluation_method:
                            # Parse verdict using standard HIRE JSON parsing
                            try:
                                v_start = content.find('"correct": ')
                                if v_start != -1:
                                    v_start += len('"correct": ')
                                    v_end = content.find(',', v_start)
                                    if v_end == -1:
                                        v_end = content.find('}', v_start)
                                    
                                    val_str = content[v_start:v_end].strip().strip('"').lower()
                                    verdict = "Yes" if val_str in ["true", "yes", "1"] else "No"
                                    
                                    # Column naming generalized for lambda, but keeping old names for base cases
                                    if folder_name == "hire_explainer_checker":
                                        verdict_col = "hire_explainer_verdict"
                                    elif folder_name == "hire_explainer_checker_query_aware":
                                        verdict_col = "hire_explainer_query_aware_verdict"
                                    elif "alignment_checker" in evaluation_method:
                                        is_obj = "obj" in evaluation_method
                                        obj_str = "_obj" if is_obj else ""
                                        
                                        variant_str = ""
                                        if "faithful" in evaluation_method: variant_str += "_faithful"
                                        elif "no_wt" in evaluation_method: variant_str += "_no_wt"
                                        
                                        if "direct_update" in evaluation_method: variant_str += "_direct_update"
                                        elif "self_refine" in evaluation_method: variant_str += "_self_refine"
                                        elif "update" in evaluation_method: variant_str += "_update"

                                        if "query_aware" in evaluation_method:
                                            base = f"hire_explainer{obj_str}_alignment_query_aware{variant_str}"
                                        else:
                                            base = f"hire_explainer{obj_str}_alignment{variant_str}"
                                            
                                        if lambda_val is not None:
                                            verdict_col = f"{base}_lambda_L{lambda_val}_verdict"
                                        else:
                                            verdict_col = f"{base}_verdict"
                                    else:
                                        verdict_col = f"{folder_name}_verdict"
                                    
                                    row[verdict_col] = verdict
                            except:
                                pass
                else:
                    row[folder_name] = None
                    if "checker" in evaluation_method:
                        if folder_name == "hire_explainer_checker":
                            verdict_col = "hire_explainer_verdict"
                        elif folder_name == "hire_explainer_checker_query_aware":
                            verdict_col = "hire_explainer_query_aware_verdict"
                        else:
                            verdict_col = f"{folder_name}_verdict"
                        row[verdict_col] = None

            elif "pseudo" in evaluation_method:
                # hire_pseudo, hire_pseudo_query_aware, hire_pseudo_checker, hire_pseudo_checker_query_aware
                
                # Construct folder name
                folder_name = evaluation_method
                if lambda_val is not None:
                    # Handle the _lambda suffix if it's missing from the method name
                    if "_lambda" not in folder_name:
                        folder_name = folder_name + "_lambda"
                    
                    folder_name = f"{folder_name}_L{lambda_val}"
                
                cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, folder_name, f"{sanitized_task_id}.json")
                
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        content = cached.get('content', '')
                        row[folder_name] = content
                        
                        if "checker" in evaluation_method:
                            # Parse verdict using standard HIRE JSON parsing
                            try:
                                v_start = content.find('"correct": ')
                                if v_start != -1:
                                    v_start += len('"correct": ')
                                    v_end = content.find(',', v_start)
                                    if v_end == -1:
                                        v_end = content.find('}', v_start)
                                    
                                    val_str = content[v_start:v_end].strip().strip('"').lower()
                                    verdict = "Yes" if val_str in ["true", "yes", "1"] else "No"
                                    
                                    # Column naming
                                    if folder_name == "hire_pseudo_checker":
                                        verdict_col = "hire_pseudo_verdict"
                                    elif folder_name == "hire_pseudo_checker_query_aware":
                                        verdict_col = "hire_pseudo_query_aware_verdict"
                                    else:
                                        verdict_col = f"{folder_name}_verdict"
                                    
                                    row[verdict_col] = verdict
                            except:
                                pass
                else:
                    row[folder_name] = None
                    if "checker" in evaluation_method:
                        if folder_name == "hire_pseudo_checker":
                            verdict_col = "hire_pseudo_verdict"
                        elif folder_name == "hire_pseudo_checker_query_aware":
                            verdict_col = "hire_pseudo_query_aware_verdict"
                        else:
                            verdict_col = f"{folder_name}_verdict"
                        row[verdict_col] = None

            elif evaluation_method.startswith('hire'):
                # HIRE Family (hire or hire_query_aware or hire_flexible)
                is_query_aware = "query_aware" in evaluation_method
                is_flexible = "flexible" in evaluation_method or "_at_most" in evaluation_method
                is_text_only = "text_only" in evaluation_method
                is_commentor = "commentor" in evaluation_method
                is_aggregator = "aggregator" in evaluation_method
                is_a2_aware = "a2_aware" in evaluation_method
                
                if is_aggregator:
                    is_text_only = True
                    is_commentor = True
                
                # Base suffix for decomposer and implementation checkers
                base_suffix = ""
                if is_query_aware:
                    base_suffix += "_query_aware"
                if is_flexible:
                    base_suffix += "_flexible"
                
                # Suffix for plan checker (includes text_only)
                plan_suffix = ""
                if is_query_aware:
                    plan_suffix += "_query_aware"
                if is_text_only:
                    plan_suffix += "_text_only"
                if is_flexible:
                    plan_suffix += "_flexible"
                
                # 1. Decomposer and Plan Checker
                decomposer_step = f'hire_decomposer{base_suffix}_N_{N}'
                plan_step = f'hire_plan_checker{plan_suffix}_N_{N}'
                
                steps_to_load = [decomposer_step, plan_step]
                
                commentor_step = ""
                if is_commentor:
                    commentor_step = f'hire_commentor_checker{base_suffix}_N_{N}'
                    steps_to_load.append(commentor_step)
                
                
                aggregator_step = ""
                agg_prefix = "hire_aggregator"
                if is_aggregator:
                    if is_a2_aware:
                        agg_prefix = "hire_aggregator_a2_aware"
                    aggregator_step = f'{agg_prefix}{base_suffix}_N_{N}'
                    steps_to_load.append(aggregator_step)
                
                for step in steps_to_load:
                    cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, step, f"{sanitized_task_id}.json")
                    
                    if os.path.exists(cache_path):
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached = json.load(f)
                        row[step] = cached.get('content', '')

                        if step == plan_step:
                            verdict_pos_start = row[step].find('"correct": ') + len('"correct": ')
                            verdict_pos_end = row[step].find(',', verdict_pos_start)
                            # Handle case where comma might not exist (e.g. last element), use closing brace
                            if verdict_pos_end == -1:
                                verdict_pos_end = row[step].find('}', verdict_pos_start)
                                
                            verdict = row[step][verdict_pos_start:verdict_pos_end].strip()
                            row[f"hire_plan{plan_suffix}_N_{N}_verdict"] = "Yes" if verdict.lower() == 'true' else "No"
                        
                        elif step == commentor_step:
                             # Similar verdict parsing for commentor checker
                             verdict_pos_start = row[step].find('"correct": ') + len('"correct": ')
                             verdict_pos_end = row[step].find(',', verdict_pos_start)
                             if verdict_pos_end == -1:
                                 verdict_pos_end = row[step].find('}', verdict_pos_start)
                                 
                             verdict = row[step][verdict_pos_start:verdict_pos_end].strip()
                             row[f"hire_commentor{base_suffix}_N_{N}_verdict"] = "Yes" if verdict.lower() == 'true' else "No"
                        
                        elif step == aggregator_step:
                             # Aggregator Verdict
                             verdict_pos_start = row[step].find('"correct": ') + len('"correct": ')
                             verdict_pos_end = row[step].find(',', verdict_pos_start)
                             if verdict_pos_end == -1:
                                 verdict_pos_end = row[step].find('}', verdict_pos_start)
                                 
                             verdict = row[step][verdict_pos_start:verdict_pos_end].strip()
                             row[f"{agg_prefix}{base_suffix}_N_{N}_verdict"] = "Yes" if verdict.lower() == 'true' else "No"

                    else:
                        row[step] = None
                
                # 2. Implementation Checkers (Stepwise)
                for checker_type in ["isolated", "context"]:
                    # checker_type "isolated" or "context"
                    # Folder: hire_implementation_checker_{checker_type}[_query_aware]_N_{N}_step_{i}
                    
                    all_correct = True
                    found_any = False
                    
                    base_prompt_type = f"hire_implementation_checker_{checker_type}{base_suffix}_N_{N}"
                    
                    for i in range(1, N + 1):
                        step_prompt_type = f"{base_prompt_type}_step_{i}"
                        col_name = f"{base_prompt_type}_step_{i}"
                        
                        cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, step_prompt_type, f"{sanitized_task_id}.json")
                        
                        if os.path.exists(cache_path):
                            found_any = True
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                cached = json.load(f)
                            
                            content = cached.get('content', '')
                            row[col_name] = content
                            
                            # Parse verdict for this step
                            try:
                                v_start = content.find('"correct": ')
                                if v_start != -1:
                                    v_start += len('"correct": ')
                                    v_end = content.find(',', v_start)
                                    if v_end == -1:
                                        v_end = content.find('}', v_start)
                                    
                                    step_verdict_str = content[v_start:v_end].strip().lower()
                                    is_step_correct = (step_verdict_str == 'true')
                                    
                                    if not is_step_correct:
                                        all_correct = False
                                else:
                                    pass
                            except:
                                pass
                        else:
                            row[col_name] = None
                            
                    # Set aggregate verdict
                    # Column name: hire_implementation_checker_{checker_type}{base_suffix}_verdict
                    verdict_col = f"hire_implementation_checker_{checker_type}{base_suffix}_N_{N}_verdict"
                    if found_any:
                        row[verdict_col] = "Yes" if all_correct else "No"
                    else:
                        row[verdict_col] = None
                
                # 3. Composite Verdicts (Plan + Implementation)
                plan_verdict = row.get(f"hire_plan{plan_suffix}_N_{N}_verdict")
                
                # Isolated Composite
                iso_verdict = row.get(f"hire_implementation_checker_isolated{base_suffix}_N_{N}_verdict")
                comp_col = f"hire_plan_imp_isolated{base_suffix}_{plan_suffix}_N_{N}_verdict"
                
                if plan_verdict is None or iso_verdict is None:
                     row[comp_col] = None
                else:
                     row[comp_col] = "Yes" if (plan_verdict == "Yes" and iso_verdict == "Yes") else "No"
                
                # Context Composite
                ctx_verdict = row.get(f"hire_implementation_checker_context{base_suffix}_N_{N}_verdict")
                comp_col_ctx = f"hire_plan_imp_context{base_suffix}_{plan_suffix}_N_{N}_verdict"
                
                if plan_verdict is None or ctx_verdict is None:
                     row[comp_col_ctx] = None
                else:
                     row[comp_col_ctx] = "Yes" if (plan_verdict == "Yes" and ctx_verdict == "Yes") else "No"
            
            elif evaluation_method.startswith("ice_"):
                # ICE Evaluation (ice_correctness, ice_usefulness)
                cache_path = os.path.join(cache_dir, dataset, code_gen_model, model_folder, evaluation_method, f"{sanitized_task_id}.json")
                
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        content = cached.get('content', '')
                        row[evaluation_method] = content
                        
                        # Parse score
                        score = content[0]#parse_score(content)
                        row[f"{evaluation_method}_score"] = score
                        
                        # Apply Verdict Logic
                        if evaluation_method == "ice_correctness":
                            # "any value less than 4 is a false"
                            # parse_score returns float. 4.0 is max.
                            # So >= 4.0 is Yes? Prompt says 0-4.
                            row[f"{evaluation_method}_verdict"] = "Yes" if score == "4" else "No"
                        elif evaluation_method == "ice_usefulness":
                             # No specific verdict logic requested, but maybe useful to have binary?
                             # For now just score is enough unless requested.
                             pass
                else:
                    row[evaluation_method] = None
                    row[f"{evaluation_method}_score"] = None
                    if evaluation_method == "ice_correctness":
                         row[f"{evaluation_method}_verdict"] = None

            else:
                pass
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def calculate_refinement_transition_metrics(original_log_path: str, refined_log_path: str) -> Dict[str, Any]:
    """
    Compares original vs refined execution logs to track test case transitions.
    
    Args:
        original_log_path: Path to the execution log of the initial code.
        refined_log_path: Path to the execution log of the refined code.
        
    Returns:
        Dictionary with counts of transitions (fail_to_pass, pass_to_fail, etc.)
    """
    if not os.path.exists(original_log_path):
        raise FileNotFoundError(f"Original execution log not found: {original_log_path}")
    if not os.path.exists(refined_log_path):
        raise FileNotFoundError(f"Refined execution log not found: {refined_log_path}")
        
    with open(original_log_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    with open(refined_log_path, 'r', encoding='utf-8') as f:
        refined_data = json.load(f)
        
    orig_results = {res['task_id']: res.get('state', []) for res in original_data.get('results', [])}
    ref_results = {res['task_id']: res.get('state', []) for res in refined_data.get('results', [])}
    
    metrics = {
        "fail_to_pass": 0,
        "pass_to_fail": 0,
        "fail_to_fail": 0,
        "pass_to_pass": 0,
        "total_orig_tests": 0,
        "total_ref_tests": 0,
        "tasks_improved": 0,
        "tasks_regressed": 0,
        "tasks_matched": 0,
        "total_original_tokens": 0,
        "total_refined_tokens": 0,
        "avg_original_tokens": 0.0,
        "avg_refined_tokens": 0.0,
        "token_reduction_rate": 0.0
    }
    
    orig_results_with_code = {res['task_id']: (res.get('state', []), res.get('generated_code', '')) for res in original_data.get('results', [])}
    ref_results_with_code = {res['task_id']: (res.get('state', []), res.get('generated_code', '')) for res in refined_data.get('results', [])}

    for task_id, (orig_state, orig_code) in orig_results_with_code.items():
        if task_id not in ref_results_with_code:
            continue
            
        ref_state, ref_code = ref_results_with_code[task_id]
        metrics["tasks_matched"] += 1
        
        # Track Tokens
        orig_tokens = count_tokens(orig_code)
        ref_tokens = count_tokens(ref_code)
        metrics["total_original_tokens"] += orig_tokens
        metrics["total_refined_tokens"] += ref_tokens

        task_f2p = 0
        task_p2f = 0
        
        # Zip compares index by index; assumes same order and count of tests per task
        for o, r in zip(orig_state, ref_state):
            metrics["total_orig_tests"] += 1
            metrics["total_ref_tests"] += 1
            if not o: # Original Fail
                if r: # Refined Pass
                    metrics["fail_to_pass"] += 1
                    task_f2p += 1
                else: # Refined Fail
                    metrics["fail_to_fail"] += 1
            else: # Original Pass
                if not r: # Refined Fail
                    metrics["pass_to_fail"] += 1
                    task_p2f += 1
                else: # Refined Pass
                    metrics["pass_to_pass"] += 1
                    
        if task_f2p > task_p2f:
            metrics["tasks_improved"] += 1
        elif task_p2f > task_f2p:
            metrics["tasks_regressed"] += 1

    # Calculate averages and reduction rate
    if metrics["tasks_matched"] > 0:
        metrics["avg_original_tokens"] = metrics["total_original_tokens"] / metrics["tasks_matched"]
        metrics["avg_refined_tokens"] = metrics["total_refined_tokens"] / metrics["tasks_matched"]
        
        if metrics["total_original_tokens"] > 0:
            metrics["token_reduction_rate"] = ((metrics["total_original_tokens"] - metrics["total_refined_tokens"]) / metrics["total_original_tokens"]) * 100
            
    return metrics
