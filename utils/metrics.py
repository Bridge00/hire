from scipy.stats import kendalltau, spearmanr
from typing import List, Dict, Any
import re
import numpy as np

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


