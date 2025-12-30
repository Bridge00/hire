import argparse
import sys
import os
from src.pipeline import PipelineRunner
from src.generator import CodeGenerator
from src.evaluator import Evaluator
from data.all_code_benchmarks import CodeData
from evaluators.py_eval import PythonEvaluator


def get_evaluator(evaluation_method: str, model_name: str, hire: bool) -> Evaluator:
    if evaluation_method == "vanilla":
        evaluator = VanillaEvaluator
    elif evaluation_method == "codejudge":
        evaluator = CodeJudgeEvaluator
    else:
        raise ValueError(f"Unknown evaluation method: {evaluation_method}")

    return evaluator(model_name, hire)
def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline.")
    
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use (e.g., 'dummy')")
    parser.add_argument("--evaluation_method", type=str, required=False, help="Base evaluation method (e.g., 'vanilla', 'codejudge')")
    parser.add_argument("--code_gen_model", type=str, required=True, help="Model to use (e.g., 'gpt-4o', 'gpt-4o-mini', 'Qwen/Qwen3-8B-Base')")
    parser.add_argument("--eval_model", type=str, required=False, help="Model to use (e.g., 'gpt-4o', 'gpt-4o-mini', 'Qwen/Qwen3-8B-Base')")
    parser.add_argument("--seed", type=int, default=95, help="Random seed for reproducibility")
    parser.add_argument("--hire", action="store_true", help="Enable HIRE (Hierarchical Reference-Free Code Evaluation)")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--no_eval", action="store_true", help="Skip LLM evaluation and only run execution tests")
    
    args = parser.parse_args()
    
    assert args.end_problem is None or args.start_problem < args.end_problem, "start_problem must be less than end_problem if end_problem is not None"
    print(f"--- Configuration ---")
    print(f"Dataset: {args.dataset}")
    print(f"LLM Eval Method: {args.evaluation_method}")
    print(f"Code Gen Model: {args.code_gen_model}")
    print(f"Eval Model: {args.eval_model}")
    print(f"HIRE Mode: {args.hire}")
    print(f"No Eval Mode: {args.no_eval}")
    print(f"---------------------")
    py_evaluator = PythonEvaluator()
    try:
        
        # 1. Load Dataset
        dataset = CodeData(args.dataset)
        print(len(dataset))
        if args.end_problem is None:
            args.end_problem = len(dataset)
        code_dataset_subset = [data for i, data in enumerate(dataset) if args.start_problem <= i < args.end_problem]
        # 2. Setup Components
        print(len(code_dataset_subset))
        generator = CodeGenerator(args.code_gen_model, dataset=args.dataset)
        evaluator = None if args.evaluation_method is None else Evaluator(args.eval_model, dataset=args.dataset, code_gen_model=args.code_gen_model)
        
        # 3. Initialize Pipeline
        runner = PipelineRunner(generator, evaluator, no_eval=args.no_eval)
        
        # 4. Run
        results = runner.run_experiment(code_dataset_subset, py_evaluator)
        
        # 5. Logging
        import subprocess
        import datetime
        import json
        
        def get_git_commit():
            try:
                return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
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
        filename = f"{log_dir}/seed_{args.seed}_{args.dataset}_{args.code_gen_model}_exec.json"

            
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
