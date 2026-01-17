from scipy.stats import kendalltau, spearmanr
from typing import List, Dict, Any
import re
import numpy as np
import pandas as pd
import json
import os
from data.all_code_benchmarks import CodeData
from utils.llm import _get_cache_key
from utils.prompts import CODEJUDGE_ANALYSIS, CODEJUDGE_SUMMARY, VANILLA_EVAL_BINARY

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


def create_results_dataframe(dataset: str, code_gen_model: str, eval_model: str = None, 
                              evaluation_method: str = None, 
                              seed: int = 95) -> pd.DataFrame:
    """
    Creates a pandas DataFrame combining execution results and LLM evaluations.
    
    Args:
        dataset: Name of the dataset (e.g., 'leetcode', 'humaneval')
        code_gen_model: Model used to generate code (e.g., 'gpt-4o-mini')
        eval_model: Model used for LLM evaluation (optional)
        evaluation_method: Evaluation method ('vanilla', 'codejudge', or None for execution-only)
        seed: Random seed used in the experiment
        
    Returns:
        DataFrame with columns: task_id, prompt, generated_code, state, feedback, pass_rate,
        and LLM evaluation columns (vanilla_eval or cj_analysis + cj_summary for CodeJudge)
    """
    # Load execution results
    exec_log_path = os.path.join(os.getenv("ROOT_DIR"), "execution_logs", f"seed_{seed}_{dataset}_{code_gen_model}_exec.json")
    
    if not os.path.exists(exec_log_path):
        raise FileNotFoundError(f"Execution log not found: {exec_log_path}")
    
    with open(exec_log_path, 'r', encoding='utf-8') as f:
        exec_data = json.load(f)
    
    
    # Build rows
    rows = []
    cache_dir = os.getenv("CACHE_DIR", ".cache")
    
    for result in exec_data['results']:
        task_id = result['task_id']
        prompt = result['prompt']
        generated_code = result['generated_code']
        state = tuple(result.get('state', []))
        feedback = result.get('feedback', '')
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
            
            if evaluation_method == 'vanilla':
                # Get vanilla evaluation from structured cache
                # Path: dataset/code_gen_model/eval_model/vanilla/task_id.json
                cache_path = os.path.join(cache_dir, dataset, code_gen_model, eval_model, evaluation_method, f"{task_id}.json")
                
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        row['vanilla'] = cached.get('content', '')
                else:
                    row['vanilla'] = None
            elif vanilla_k_match:
                k = int(vanilla_k_match.group(1))
                scores = []
                for j in range(1, k + 1):
                    # Cache prompt type used in generate_batch.py was f"vanilla_k{j}"
                    cache_path = os.path.join(cache_dir, dataset, code_gen_model, eval_model, f"vanilla_k{j}", f"{task_id}.json")
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
                    cache_path = os.path.join(cache_dir, dataset, code_gen_model, eval_model, step, f"{task_id}.json")
                    
                    if os.path.exists(cache_path):
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached = json.load(f)
                            row[step] = cached.get('content', '')
                    else:
                        row[step] = None
                

            elif evaluation_method == 'hire':
                # Get Hire Decomposer analysis
                # Path: dataset/code_gen_model/eval_model/hire_decomposer/task_id.json
                for step in ['hire_decomposer', 'hire_plan_checker']:
                    cache_path = os.path.join(cache_dir, dataset, code_gen_model, eval_model, step, f"{task_id}.json")
                    
                    if os.path.exists(cache_path):
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cached = json.load(f)
                        row[step] = cached.get('content', '')

                        if step == 'hire_plan_checker':
                            
                            verdict_pos_start = row[step].find('"correct": ') + len('"correct": ')
                            verdict_pos_end = row[step].find(',', verdict_pos_start)
                            
                            verdict = row[step][verdict_pos_start:verdict_pos_end]
                            row["hire_plan_verdict"] = "Yes" if verdict.lower() == 'true' else "No"
                    else:
                        row[step] = None
            
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
        "tasks_matched": 0
    }
    
    for task_id, orig_state in orig_results.items():
        if task_id not in ref_results:
            continue
            
        ref_state = ref_results[task_id]
        metrics["tasks_matched"] += 1
        
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
            
    return metrics
