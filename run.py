import argparse
import sys
import os
from src.pipeline import PipelineRunner
from src.pipeline import PipelineRunner
from data.all_code_benchmarks import CodeData
from evaluators.py_eval import PythonEvaluator
from evaluators import get_evaluator


def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline.")
    
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use (e.g., 'dummy')")
    parser.add_argument("--evaluation_method", type=str, required=False, help="Base evaluation method (e.g., 'vanilla', 'codejudge')")
    parser.add_argument("--code_gen_model", type=str, required=False, help="Model to use (e.g., 'gpt-4o', 'gpt-4o-mini', 'Qwen/Qwen3-8B-Base')")
    parser.add_argument("--eval_model", type=str, required=False, help="Model to use (e.g., 'gpt-4o', 'gpt-4o-mini', 'Qwen/Qwen3-8B-Base')")
    parser.add_argument("--seed", type=int, default=95, help="Random seed for reproducibility")

    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--no_eval", action="store_true", help="Skip LLM evaluation and only run execution tests")
    parser.add_argument("--execute_solution", action="store_true", help="Execute canonical solution instead of generated code")
    parser.add_argument("--eval_source", type=str, default=None, help="Source key to evaluate (e.g., 'incorrect_solution', 'misleading_task'). Overrides execute_solution code selection.")
    parser.add_argument("--force", action="store_true", help="Force re-generation of LLM responses (bypass cache)")
    parser.add_argument("--parallel", action="store_true", help="Run benchmarks in parallel")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for parallel execution")
    
    # Refinement arguments
    parser.add_argument("--refine_results", action="store_true", help="Run refined code from cache")
    parser.add_argument("--refine_model", type=str, help="Model used for refinement")
    parser.add_argument("--eval_prompt", type=str, help="Evaluation prompt used for refinement")
    parser.add_argument("--n", type=int, default=3, help="N parameter for HIRE prompts")
    parser.add_argument("--k", type=int, default=1, help="K parameter for vanilla prompts")
   
    args = parser.parse_args()
    
    assert args.end_problem is None or args.start_problem < args.end_problem, "start_problem must be less than end_problem if end_problem is not None"
    print(f"--- Configuration ---")
    print(f"Dataset: {args.dataset}")
    print(f"LLM Eval Method: {args.evaluation_method}")
    print(f"Code Gen Model: {args.code_gen_model}")
    print(f"Eval Model: {args.eval_model}")
    print(f"No Eval Mode: {args.no_eval}")
    if args.refine_results:
        print(f"Refine Mode: Active")
        print(f"Refine Model: {args.refine_model}")
        print(f"Refine Eval Source: {args.eval_source}")
        print(f"Refine Eval Prompt: {args.eval_prompt}")
    print(f"---------------------")
    if args.refine_results:
        if not all([args.eval_source, args.eval_prompt, args.eval_model]):
             print("Error: --refine_results requires --eval_source, --eval_prompt, and --eval_model.")
             sys.exit(1)
        
        # Construct specialized code_gen_model name for cache resolution
        refine_model = args.refine_model or args.code_gen_model or "gpt-4o-mini"
        k_suffix = f"_k{args.k}" if (args.eval_prompt == "vanilla" and args.k > 1) else ""
        n_suffix = f"_N_{args.n}" if (args.eval_prompt and args.eval_prompt.startswith("hire_") and args.eval_prompt not in ["hire_explainer", "hire_explainer_query_aware", "hire_explainer_checker", "hire_explainer_checker_query_aware"]) else ""
        specialized_gen_model = f"{refine_model}_refine_{args.eval_source}_{args.eval_prompt}{n_suffix}{k_suffix}_{args.eval_model}"
        print(f"Constructed specialized model name for cache: {specialized_gen_model}")
        args.code_gen_model = specialized_gen_model

    py_evaluator = get_evaluator(args.dataset)
    try:
        # 1. Load Dataset
        dataset = CodeData(args.dataset)
        print(len(dataset))
        if args.end_problem is None:
            args.end_problem = len(dataset)
        code_dataset_subset = [data for i, data in enumerate(dataset) if args.start_problem <= i < args.end_problem]

        # 2. Setup Components
        print(len(code_dataset_subset))
        
        # 3. Initialize Pipeline
        # If refine_results is set, we use the specialized model name (which hits refinement cache).
        # We pass eval_source=None to the runner so it doesn't try to pull raw code from dataset.
        runner_eval_source = args.eval_source if not args.refine_results else None
        
        runner = PipelineRunner(
            code_gen_model=args.code_gen_model, 
            eval_model=args.eval_model, 
            eval_prompt_type=args.evaluation_method, 
            no_eval=args.no_eval, 
            execute_solution=args.execute_solution, 
            dataset_name=args.dataset, 
            eval_source=runner_eval_source
        )
       
        # 4. Run
        results = runner.run_experiment(code_dataset_subset, py_evaluator, parallel=args.parallel, num_workers=args.num_workers)
        
        # 5. Logging
        import subprocess
        import datetime
        import json
        
        def get_git_commit():
            try:
                # Use a more cross-platform way to get git commit or return unknown
                return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            except Exception:
                return "unknown"

        log_data = {
            "config": vars(args),
            "git_commit": get_git_commit(),
            "timestamp": datetime.datetime.now().isoformat(),
            "results": results
        }
        
        # Create logs directory
        log_dir = "execution_logs"
        # Naming convention for execution logs
        range_suffix = f"_{args.start_problem}_{args.end_problem}"
        
        if args.refine_results:
             filename = f"{log_dir}/{args.dataset}_{args.code_gen_model}_exec{range_suffix}.json"
        elif args.eval_source:
             filename = f"{log_dir}/{args.dataset}_{args.eval_source}_exec{range_suffix}.json"
        elif args.execute_solution:
            filename = f"{log_dir}/{args.dataset}_solutions_exec{range_suffix}.json"
        else:
            filename = f"{log_dir}/seed_{args.seed}_{args.dataset}_{args.code_gen_model}_exec{range_suffix}.json"
            
        os.makedirs(log_dir, exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)
            
        print(f"\nCompleted {len(results)} items.")
        print(f"Results saved to: {filename}")

        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
