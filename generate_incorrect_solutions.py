
import argparse
import json
import os
import sys
from data.all_code_benchmarks import CodeData

# Define prompt here since utils/prompts.py edit was difficult
INCORRECT_SOLUTION_PROMPT = """You are an expert software engineer.
Your task is to take a correct solution to a coding problem and introduce a subtle bug so that it fails some test cases.
The perturbations should be subtle and not too complex, such as changing <= to <, == to !=, or forgetting to increment values. 
Problem Description:
{PROBLEM}

Correct Solution:
{CODE}

Tests Passed (by the correct solution):
{TESTS}

Please provide the incorrect solution wrapped in a code block. Do not provide explanation.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="Path to execution log (required if source=log)")
    parser.add_argument("--dataset", type=str, default="leetcode", help="Dataset name")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save updated dataset")
    parser.add_argument("--batch_output_dir", type=str, default="batch_jobs_perturb", help="Dir for batch jobs")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model for perturbation")
    parser.add_argument("--source", type=str, default="log", choices=["log", "canonical"], help="Source for code to perturb: 'log' (execution results) or 'canonical' (dataset canonical solution)")
    args = parser.parse_args()

    results_map = {}
    if args.source == "log":
        if not args.log_file:
             print("Error: --log_file is required when --source is 'log'")
             sys.exit(1)
             
        print(f"Reading log file: {args.log_file}")
        try:
            with open(args.log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            results_list = log_data.get('results', [])
            results_map = {item['task_id']: item for item in results_list}
            print(f"Loaded {len(results_map)} results from log.")
        except Exception as e:
            print(f"Error reading log file: {e}")
            sys.exit(1)
    
    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = CodeData(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    updated_items = []
    batch_requests = []
    
    os.makedirs(args.batch_output_dir, exist_ok=True)
    batch_filename = os.path.join(args.batch_output_dir, f"{args.dataset}_perturb_incorrect_{args.model}.jsonl")

    incorrect_count = 0
    perturb_count = 0

    # Iterate through dataset to preserve order/structure
    for i in range(len(dataset)):
        # Access raw item
        item = dataset.dataset[i]
        task_id = item.get('task_id')
        
        # In case task_id key is missing or different, logic above handles it via CodeData usually
        if not task_id:
            # Try to infer or skip
            print(f"Warning: Item {i} has no task_id")
            updated_items.append(item)
            continue
            
        code_to_perturb = None
        feedback_str = "Tests passed by canonical solution."

        if args.source == "log":
             # Sanitize task_id for lookup (run.py/CodeData converts / to _)
            lookup_id = task_id.replace('/', '_')
                
            if lookup_id not in results_map:
                # If not in the log, just keep as is
                updated_items.append(item)
                continue
                
            result = results_map[lookup_id]
            pass_rate = result.get('pass_rate', 0.0)
            
            if pass_rate < 1.0:
                # Case 1: Already incorrect
                # Use 'generated_code' from log as 'incorrect_solution'
                code = result.get('generated_code', "")
                if code:
                    item['incorrect_solution'] = code
                    incorrect_count += 1
                updated_items.append(item)
                continue # Skip perturbation
                
            # Case 2: Correct (pass_rate == 1.0)
            code_to_perturb = result.get('generated_code', "")
            feedback_str = result.get('feedback', "")
        
        else: # source == "canonical"
             code_to_perturb = item.get('canonical_solution', "")
             # For canonical, we don't have execution feedback, but we can pass the raw tests or a default message
             # Passing raw tests is safer for context
             tests_raw = item.get('test', "")
             if isinstance(tests_raw, list):
                 feedback_str = "\n".join(tests_raw)
             else:
                 feedback_str = str(tests_raw)

        # Generate perturbation prompt
        problem = item.get('prompt', "") # Description
        
        if not code_to_perturb or not problem:
            print(f"Warning: Missing code or problem for task {task_id}, skipping perturbation.")
            updated_items.append(item)
            continue

        prompt_text = INCORRECT_SOLUTION_PROMPT.format(
            PROBLEM=problem,
            CODE=code_to_perturb,
            TESTS=feedback_str
        )
        
        # Create batch request
        request_id = task_id # Use task_id as custom_id to easily map back
        req = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt_text}
                ]
            }
        }
        batch_requests.append(req)
        perturb_count += 1
        # Do NOT add to updated_items; these will be added after perturbation is processed.

    # Save updated dataset (with found incorrect solutions from log, or just existing items if canonical mode didn't add any yet)
    print(f"Saving updated dataset to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in updated_items:
            f.write(json.dumps(item) + '\n')
            
    print(f"Saved dataset. Added 'incorrect_solution' to {incorrect_count} items (from log).")
    
    # Save batch file
    if batch_requests:
        with open(batch_filename, 'w', encoding='utf-8') as f:
            for req in batch_requests:
                f.write(json.dumps(req) + '\n')
        print(f"Generated batch file with {len(batch_requests)} requests: {batch_filename}")
        print("Run this batch file to generate incorrect solutions.")
        print(f"After running batch, you will need to process the results and merge them into '{args.output_file}'.")
    else:
        print("No perturbation requests generated.")

if __name__ == "__main__":
    main()
