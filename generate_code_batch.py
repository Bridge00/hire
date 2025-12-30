import argparse
import json
import os
import sys
from dotenv import load_dotenv

from data.all_code_benchmarks import CodeData
from utils.prompts import CODEGEN_SYS

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API requests.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'leetcode')")
    parser.add_argument("--code_gen_model", type=str, required=True, help="Model (e.g., 'gpt-4o-mini')")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--output_dir", type=str, default="batch_jobs", help="Directory to save batch files")

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = CodeData(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.end_problem is None:
        args.end_problem = len(dataset)
    
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Processing items from {start} to {end}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.code_gen_model}_{start}_{end}.jsonl")
    
    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    
    requests_created = 0
    skipped_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                # CodeData __getitem__ returns: task_id, prompt, test, canonical_solution
                task_id, prompt, _, _ = dataset[i]
                
                # Check structured cache: dataset/code_gen_model/task_id.json
                cache_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, f"{task_id}.json")
                
                if os.path.exists(cache_path):
                    skipped_count += 1
                    continue
                
                # Create Batch Request
                # Use task_id as custom_id for structured cache
                request_body = {
                    "custom_id": task_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.code_gen_model,
                        "messages": [
                            {"role": "system", "content": CODEGEN_SYS},
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

if __name__ == "__main__":
    main()
