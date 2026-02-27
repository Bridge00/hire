import argparse
import json
import os
import glob
import sys

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from execution logs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--code_gen_model", type=str, help="Base model name")
    parser.add_argument("--eval_source", type=str, help="Evaluation source (for refinement or specific keys)")
    parser.add_argument("--eval_prompt", type=str, help="Evaluation prompt used")
    parser.add_argument("--eval_model", type=str, help="Evaluation model used")
    parser.add_argument("--n", type=int, default=3, help="N parameter for HIRE prompts")
    parser.add_argument("--k", type=int, default=1, help="K parameter for vanilla prompts")
    parser.add_argument("--refine_results", action="store_true", help="Calculate metrics for refined results")
    parser.add_argument("--refine_model", type=str, help="Model used for refinement")

    args = parser.parse_args()

    model_id = args.code_gen_model
    if args.refine_results:
        refine_model = args.refine_model or args.code_gen_model or "gpt-4o-mini"
        k_suffix = f"_k{args.k}" if (args.eval_prompt == "vanilla" and args.k > 1) else ""
        n_suffix = f"_N_{args.n}" if (args.eval_prompt and args.eval_prompt.startswith("hire_") and args.eval_prompt not in ["hire_explainer", "hire_explainer_query_aware", "hire_explainer_checker", "hire_explainer_checker_query_aware"]) else ""
        model_id = f"{refine_model}_refine_{args.eval_source}_{args.eval_prompt}{n_suffix}{k_suffix}_{args.eval_model}"
    elif args.eval_source:
        model_id = args.eval_source

    log_dir = "execution_logs"
    # Search for all shards matching the prefix
    # Format: {dataset}_{model_id}_exec_{start}_{end}.json or seed_..._{dataset}_{model_id}_exec...
    pattern_refine = f"{log_dir}/{args.dataset}_{model_id}_exec_*.json"
    pattern_source = f"{log_dir}/{args.dataset}_{args.eval_source}_exec_*.json" if args.eval_source and not args.refine_results else None
    pattern_standard = f"{log_dir}/seed_*_{args.dataset}_{model_id}_exec_*.json"

    files = []
    if args.refine_results:
        files = glob.glob(pattern_refine)
    elif args.eval_source:
        files = glob.glob(pattern_source)
    else:
        files = glob.glob(pattern_standard)

    if not files:
        print(f"No log files found for pattern matching: {model_id}")
        return

    print(f"Found {len(files)} log files. Aggregating results...")

    total_tasks = 0
    passed_tasks = 0
    processed_task_ids = set()

    for file in sorted(files):
        with open(file, "r") as f:
            try:
                data = json.load(f)
                results = data.get("results", [])
                for res in results:
                    task_id = res.get("task_id")
                    if task_id in processed_task_ids:
                        continue
                    
                    processed_task_ids.add(task_id)
                    total_tasks += 1
                    pass_rate = res.get("pass_rate", 0.0)
                    if pass_rate == 1.0:
                        passed_tasks += 1
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if total_tasks == 0:
        print("No tasks found in log files.")
        return

    success_rate = (passed_tasks / total_tasks) * 100
    print("\n--- Metrics ---")
    print(f"Dataset:      {args.dataset}")
    print(f"Model ID:     {model_id}")
    print(f"Total Tasks:  {total_tasks}")
    print(f"Passed Tasks: {passed_tasks} (Pass@1)")
    print(f"Success Rate: {success_rate:.2f}%")
    print("----------------")

if __name__ == "__main__":
    main()
