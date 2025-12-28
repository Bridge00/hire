import argparse
import sys
import os
from src.pipeline import PipelineRunner
from src.generator import CodeGenerator
from src.evaluator import Evaluator

def get_dataset(dataset_name: str) -> List[Dict[str, str]]:
    pass

def get_generator(model_name: str) -> CodeGenerator:
    return CodeGenerator(model_name)

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
    
    args = parser.parse_args()
    
    print(f"--- Configuration ---")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.evaluation_method}")
    print(f"Code Gen Model: {args.code_gen_model}")
    print(f"Eval Model: {args.eval_model}")
    print(f"HIRE Mode: {args.hire}")
    print(f"---------------------")

    try:
        # 1. Load Dataset
        dataset = get_dataset(args.dataset)
        
        # 2. Setup Components
        generator = CodeGenerator(args.code_gen_model)
        evaluator = None if args.evaluation_method is None else Evaluator(args.eval_model, args.evaluation_method, args.hire)
        
        # 3. Initialize Pipeline
        runner = PipelineRunner(generator, evaluator)
        
        # 4. Run
        results = runner.run_experiment(dataset)

        if evaluator is None:
            return
        
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
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate filename
        filename = f"{log_dir}/seed_{args.seed}_{args.dataset}_{args.evaluation_method}.json"
        
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2)
            
        print(f"\nCompleted {len(results)} items.")
        print(f"Results saved to: {filename}")
        if results:
            print(f"Sample Result: {results[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
