import argparse
import json
import os
import sys
from dotenv import load_dotenv

from data.all_code_benchmarks import CodeData
from utils.prompts import CODEGEN_SYS, VANILLA_EVAL_BINARY, CODEJUDGE_ANALYSIS, CODEJUDGE_SUMMARY
from utils.llm import _get_cache_key, clean_code

load_dotenv()

# From TextGrad
EVAL_SYS = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise critiques."

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API requests for Evaluation.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'leetcode')")
    parser.add_argument("--code_gen_model", type=str, required=True, help="Model used to generate the code (e.g., 'gpt-4o-mini')")
    parser.add_argument("--eval_model", type=str, required=True, help="Model to use for evaluation (e.g., 'gpt-4o')")
    parser.add_argument("--prompt", type=str, default="vanilla", choices=["vanilla", "cj_analysis", "cj_summ"], help="Evaluation prompt to use")
    parser.add_argument("--analysis_model", type=str, default=None, help="Model used for analysis (only for cj_summ). Defaults to eval_model.")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--output_dir", type=str, default="batch_jobs_eval", help="Directory to save batch files")

    args = parser.parse_args()

    # Default analysis_model to eval_model if not provided
    if args.analysis_model is None:
        args.analysis_model = args.eval_model

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
    
    print(f"Processing items from {start} to {end} with prompt '{args.prompt}'")

    os.makedirs(args.output_dir, exist_ok=True)
    # Updated filename format as requested
    output_file = os.path.join(
        args.output_dir, 
        f"{args.dataset}_code_model_{args.code_gen_model}_{args.prompt}_eval_eval_{args.eval_model}_{start}_{end}.jsonl"
    )
    
    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    
    requests_created = 0
    skipped_count = 0
    missing_code_count = 0
    missing_analysis_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                # CodeData __getitem__ returns: task_id, prompt, test, canonical_solution
                task_id, problem_prompt, _, _ = dataset[i]
                
                # 1. Locate the generated code in cache
                gen_cache_key = _get_cache_key(CODEGEN_SYS, problem_prompt, args.code_gen_model)
                gen_cache_path = os.path.join(cache_dir, f"{gen_cache_key}.json")
                
                if not os.path.exists(gen_cache_path):
                    missing_code_count += 1
                    continue
                
                with open(gen_cache_path, 'r', encoding='utf-8') as f:
                    gen_data = json.load(f)
                    raw_code = gen_data.get("content", "")

                cleaned_code = clean_code(raw_code)

                # 2. Construct Evaluation Prompt
                eval_user_prompt = ""
                
                if args.prompt == "vanilla":
                    eval_user_prompt = VANILLA_EVAL_BINARY.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                elif args.prompt == "cj_analysis":
                    eval_user_prompt = CODEJUDGE_ANALYSIS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                elif args.prompt == "cj_summ":
                    # For summary, we need the stored analysis result
                    # First, reconstruct the analysis key
                    analysis_prompt = CODEJUDGE_ANALYSIS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                    analysis_key = _get_cache_key(EVAL_SYS, analysis_prompt, args.analysis_model)
                    analysis_path = os.path.join(cache_dir, f"{analysis_key}.json")
                    
                    if not os.path.exists(analysis_path):
                        missing_analysis_count += 1
                        continue
                        
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                        analysis_content = analysis_data.get("content", "")
                        
                    eval_user_prompt = CODEJUDGE_SUMMARY.format(ANALYSIS=analysis_content)

                if not eval_user_prompt:
                    print(f"Error: Empty prompt for item {i}")
                    continue

                # 3. Check if Evaluation is already cached
                eval_cache_key = _get_cache_key(EVAL_SYS, eval_user_prompt, args.eval_model)
                eval_cache_path = os.path.join(cache_dir, f"{eval_cache_key}.json")

                if os.path.exists(eval_cache_path):
                    skipped_count += 1
                    continue

                # 4. Create Batch Request
                request_body = {
                    "custom_id": eval_cache_key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.eval_model,
                        "messages": [
                            {"role": "system", "content": EVAL_SYS},
                            {"role": "user", "content": eval_user_prompt}
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
    print(f"Skipped (already cached): {skipped_count}")
    print(f"Skipped (missing generated code): {missing_code_count}")
    if args.prompt == "cj_summ":
        print(f"Skipped (missing analysis): {missing_analysis_count}")

if __name__ == "__main__":
    main()
