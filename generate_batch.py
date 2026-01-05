import argparse
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

from data.all_code_benchmarks import CodeData
#from utils.prompts import CODEGEN_SYS, VANILLA_EVAL_BINARY, CODEJUDGE_ANALYSIS, CODEJUDGE_SUMMARY, HIRE_DECOMPOSER
import utils.prompts as up
from utils.llm import clean_code

load_dotenv()

# From TextGrad
EVAL_SYS = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise critiques."

def generate_code_batch(args, dataset, cache_dir):
    """Generate batch requests for code generation."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating CODE GENERATION batch for items {start} to {end}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.code_gen_model}_{start}_{end}.jsonl")
    
    requests_created = 0
    skipped_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, prompt, _, _ = dataset[i]
                
                # Check structured cache: dataset/code_gen_model/task_id.json
                cache_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, f"{task_id}.json")
                
                if os.path.exists(cache_path):
                    skipped_count += 1
                    continue
                
                # Create Batch Request
                request_body = {
                    "custom_id": task_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.code_gen_model,
                        "messages": [
                            {"role": "system", "content": up.CODEGEN_SYS},
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
    
    return output_file

def generate_eval_batch(args, dataset, cache_dir):
    """Generate batch requests for evaluation."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating EVALUATION batch for items {start} to {end} with prompt '{args.eval_prompt}'")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, 
        f"{args.dataset}_code_model_{args.code_gen_model}_{args.eval_prompt}_eval_eval_{args.eval_model}_{start}_{end}.jsonl"
    )
    
    requests_created = 0
    skipped_count = 0
    missing_code_count = 0
    missing_analysis_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                task_id, problem_prompt, _, _ = dataset[i]
                
                # 1. Locate the generated code in structured cache
                gen_cache_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, f"{task_id}.json")
                
                if not os.path.exists(gen_cache_path):
                    missing_code_count += 1
                    continue
                
                with open(gen_cache_path, 'r', encoding='utf-8') as f:
                    gen_data = json.load(f)
                    raw_code = gen_data.get("content", "")

                cleaned_code = clean_code(raw_code)

                # 2. Construct Evaluation Prompt
                eval_user_prompt = ""
                eval_prompt_type = args.eval_prompt
                
                if args.eval_prompt == "vanilla":
                    eval_user_prompt =up.VANILLA_EVAL_BINARY.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                elif args.eval_prompt == "hire_decomposer":
                    eval_user_prompt = up.HIRE_DECOMPOSER.format(N=args.n, CODE=cleaned_code)
                    eval_prompt_type = "hire_decomposer"
                elif args.eval_prompt == "hire_plan_checker":

    
                    decomposed_plan_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, args.eval_model, "hire_decomposer", f"{task_id}.json")
                    
                    if not os.path.exists(decomposed_plan_path):
                        missing_code_count += 1
                        continue
                    
                    with open(decomposed_plan_path, 'r', encoding='utf-8') as f:
                        plan_data = json.load(f)
                        plan = plan_data.get("content", "")
                    
                    eval_user_prompt = up.HIRE_PLAN_CHECKER.format(PROBLEM=problem_prompt, plan=plan)
                    eval_prompt_type = "hire_plan_checker"

                elif args.eval_prompt == "cj_analysis":
                    eval_user_prompt = up.CODEJUDGE_ANALYSIS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
                elif args.eval_prompt == "cj_summ":
                    # Check structured cache for analysis
                    analysis_model = args.analysis_model or args.eval_model
                    analysis_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, analysis_model, "cj_analysis", f"{task_id}.json")
                    
                    if not os.path.exists(analysis_path):
                        missing_analysis_count += 1
                        continue
                        
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                        analysis_content = analysis_data.get("content", "")
                        
                    eval_user_prompt = up.CODEJUDGE_SUMMARY.format(ANALYSIS=analysis_content)
                    eval_prompt_type = "cj_summary"

                if not eval_user_prompt:
                    print(f"Error: Empty prompt for item {i}")
                    continue

                # 3. Check if Evaluation is already cached
                eval_cache_path = os.path.join(cache_dir, args.dataset, args.code_gen_model, args.eval_model, eval_prompt_type, f"{task_id}.json")

                if os.path.exists(eval_cache_path):
                    skipped_count += 1
                    continue

                # 4. Create Batch Request
                request_body = {
                    "custom_id": task_id,
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
    if args.eval_prompt == "cj_summ":
        print(f"Skipped (missing analysis): {missing_analysis_count}")
    
    return output_file

def submit_batch(batch_file):
    """Submit a batch file to OpenAI API."""
    if not os.path.exists(batch_file):
        print(f"Error: Batch file {batch_file} does not exist.")
        sys.exit(1)
    
    try:
        client = OpenAI()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file or environment variables.")
        sys.exit(1)
    
    print(f"Uploading file: {batch_file}")
    
    try:
        batch_input_file = client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        file_id = batch_input_file.id
        print(f"File uploaded. ID: {file_id}")
        
        print("Creating batch job...")
        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Batch job for {os.path.basename(batch_file)}"
            }
        )
        
        job_id = batch_job.id
        print(f"Batch job created. ID: {job_id}")
        
        # Save response
        output_dir = "batch_job_metadata"
        os.makedirs(output_dir, exist_ok=True)
        
        input_basename = os.path.basename(batch_file)
        # Remove extension for cleaner name
        if input_basename.endswith('.jsonl'):
            input_basename = input_basename[:-6]
        
        output_file = os.path.join(output_dir, f"{input_basename}_job.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # batch_job is a Pydantic model in recent SDKs
            if hasattr(batch_job, 'model_dump_json'):
                f.write(batch_job.model_dump_json(indent=2))
            elif hasattr(batch_job, 'to_json'):
                f.write(json.dumps(batch_job.to_json(), indent=2))
            else:
                try:
                    f.write(json.dumps(batch_job.__dict__, indent=2, default=str))
                except:
                    f.write(str(batch_job))
        
        print(f"Job details saved to: {output_file}")
        print(f"\nTo check status later, run: python download_batch.py --job_file {output_file}")
        
    except Exception as e:
        print(f"Error submitting batch: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API requests for code generation or evaluation.")
    parser.add_argument("--mode", type=str, required=True, choices=["code", "eval"], 
                        help="Mode: 'code' for code generation, 'eval' for evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'leetcode')")
    parser.add_argument("--code_gen_model", type=str, required=True, help="Model for code generation (e.g., 'gpt-4o-mini')")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save batch files (default: batch_jobs for code, batch_jobs_eval for eval)")
    
    # Evaluation-specific arguments
    parser.add_argument("--eval_model", type=str, help="Model for evaluation (required if mode=eval)")
    parser.add_argument("--eval_prompt", type=str, choices=["vanilla", "cj_analysis", "cj_summ", "hire_decomposer"], 
                        help="Evaluation prompt type (required if mode=eval)")
    parser.add_argument("--analysis_model", type=str, help="Model used for analysis (only for cj_summ, defaults to eval_model)")
    parser.add_argument("--n", type=int, default=3, help="Number of steps for hire_decomposer")
    
    # Submission control
    parser.add_argument("--dont_submit", action="store_true", help="Don't submit batch to OpenAI (only generate the file)")

    args = parser.parse_args()
    
    # Set default output directory based on mode
    if args.output_dir is None:
        args.output_dir = "batch_jobs" if args.mode == "code" else "batch_jobs_eval"
    
    # Validate mode-specific arguments
    if args.mode == "eval":
        if not args.eval_model or not args.eval_prompt:
            print("Error: --eval_model and --eval_prompt are required when mode=eval")
            sys.exit(1)
        if args.eval_prompt == "cj_summ" and not args.analysis_model:
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
    
    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    
    # Generate batch based on mode
    if args.mode == "code":
        batch_file = generate_code_batch(args, dataset, cache_dir)
    else:  # eval
        batch_file = generate_eval_batch(args, dataset, cache_dir)
    
    # Submit batch if not disabled
    if not args.dont_submit:
        print("\n" + "="*50)
        print("Submitting batch to OpenAI...")
        print("="*50)
        submit_batch(batch_file)
    else:
        print("\nBatch file created but not submitted (--dont_submit flag set)")
        print(f"To submit later, run: python run_batch.py --batch_input_file {batch_file}")

if __name__ == "__main__":
    main()
