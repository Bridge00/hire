import argparse
import sys
import os
from src.pipeline import PipelineRunner

def main():
    parser = argparse.ArgumentParser(description="Run experiment pipeline.")
    
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use (e.g., 'dummy')")
    parser.add_argument("--evaluation_method", type=str, required=True, help="Base evaluation method (e.g., 'vanilla')")
    parser.add_argument("--hire", action="store_true", help="Enable HIRE (Hierarchical Reference-Free Code Evaluation)")
    
    args = parser.parse_args()
    
    print(f"--- Configuration ---")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.evaluation_method}")
    print(f"HIRE Mode: {args.hire}")
    print(f"---------------------")

    try:
        # 1. Load Dataset
        dataset = get_dataset(args.dataset)
        
        # 2. Setup Components
        generator = get_generator()
        evaluator = get_evaluator(args.evaluation_method, args.hire)
        
        # 3. Initialize Pipeline
        runner = PipelineRunner(generator, evaluator)
        
        # 4. Run
        results = runner.run_experiment(dataset)
        
        # 5. Output
        print(f"\nCompleted {len(results)} items.")
        if results:
            print(f"Sample Result: {results[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
