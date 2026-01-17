import argparse
import json
import os
from utils.metrics import calculate_refinement_transition_metrics

def main():
    parser = argparse.ArgumentParser(description="Calculate refinement transition metrics between two execution logs.")
    parser.add_argument("--original", type=str, required=True, help="Path to the original execution log")
    parser.add_argument("--refined", type=str, required=True, help="Path to the refined execution log")
    
    args = parser.parse_args()
    
    try:
        metrics = calculate_refinement_transition_metrics(args.original, args.refined)
        
        print("\n" + "="*40)
        print("REFINEMENT TRANSITION METRICS")
        print("="*40)
        print(f"Original Log: {os.path.basename(args.original)}")
        print(f"Refined Log:  {os.path.basename(args.refined)}")
        print("-" * 40)
        print(f"Tasks Matched:      {metrics['tasks_matched']}")
        print(f"Total Test Cases:   {metrics['total_orig_tests']}")
        print("-" * 40)
        print(f"FAIL -> PASS:       {metrics['fail_to_pass']} (Improved)")
        print(f"PASS -> FAIL:       {metrics['pass_to_fail']} (Regressed)")
        print(f"FAIL -> FAIL:       {metrics['fail_to_fail']} (No change)")
        print(f"PASS -> PASS:       {metrics['pass_to_pass']} (No change)")
        print("-" * 40)
        print(f"Tasks Improved:     {metrics['tasks_improved']}")
        print(f"Tasks Regressed:    {metrics['tasks_regressed']}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
