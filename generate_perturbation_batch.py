import argparse
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

from data.all_code_benchmarks import CodeData
from utils.prompts import (
    MISLEADING_TASK_BIAS_PROMPT,
    REVERSE_AUTHORITY_BIAS_PROMPT,
    AUTHORITY_BIAS_PROMPT
)

load_dotenv()

def create_perturbation_prompt(code: str, language: str, problem_description: str, perturbation_type: str) -> str:
    """Create a prompt for the LLM to perturb the solution based on the specified type."""
    
    # Capitalize first letter for display (e.g. "Python")
    program_language = language.capitalize()
    if language == "cpp":
        program_language = "C++"
    
    if perturbation_type == "misleading_task":
        prompt = MISLEADING_TASK_BIAS_PROMPT.format(
            PROBLEM=problem_description,
            CODE=code,
            PROGRAM_LANGUAGE=program_language,
            PROGRAM_LANGUAGE_LOWER=language
        )
    elif perturbation_type == "reverse_authority":
        prompt = REVERSE_AUTHORITY_BIAS_PROMPT.format(
            PROBLEM=problem_description, # Note: Prompt template uses PROBLEM but logic above didn't show it? Checking prompts.py content...
            # Actually REVERSE_AUTHORITY_BIAS_PROMPT in prompts.py has {PROBLEM} and {CODE}.
            CODE=code,
            PROGRAM_LANGUAGE=program_language,
            PROGRAM_LANGUAGE_LOWER=language
        )
    elif perturbation_type == "authority":
        prompt = AUTHORITY_BIAS_PROMPT.format(
            PROBLEM=problem_description,
            CODE=code,
            PROGRAM_LANGUAGE=program_language,
            PROGRAM_LANGUAGE_LOWER=language
        )
    else: # default
        prompt = f"""You are tasked with adding misleading comments to correct code. Your goal is to add comments that suggest the code is incorrect or has issues, even though the code itself is actually correct and functional.

{code}    
4. Make the comments seem like they're questioning the correctness
5. Do NOT add comments that explicitly say "this is wrong" - be more subtle
6. Return ONLY the code with added comments, no explanation

Perturbed code:"""
    
    return prompt

def generate_perturbation_batch(args, dataset):
    """Generate batch requests for code perturbation."""
    start = args.start_problem
    end = min(args.end_problem, len(dataset))
    
    print(f"Generating PERTURBATION batch for dataset '{args.dataset}' items {start} to {end}")
    
    # Determine effective source for filename
    effective_source = args.code_source
    if args.perturbation_type == "reverse_authority":
        effective_source = "canonical"
    elif args.perturbation_type == "authority":
        effective_source = "incorrect"

    print(f"Type: {args.perturbation_type}, Source: {effective_source}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{effective_source}" if effective_source != "canonical" else ""
    if args.perturbation_type != "default":
        suffix += f"_{args.perturbation_type}"
    
    output_file = os.path.join(args.output_dir, f"{args.dataset}_perturbation_{args.model}{suffix}_{start}_{end}.jsonl")
    
    requests_created = 0
    
    # Determine programming language
    if args.dataset in ["leetcode", "humaneval_py"]:
        language = "python"
    elif args.dataset == "humaneval_js":
        language = "javascript"
    elif args.dataset == "humaneval_java":
        language = "java"
    elif args.dataset == "humaneval_cpp":
        language = "cpp"
    elif args.dataset == "humaneval_go":
        language = "go"
    else:
        # Fallback to python if unknown, or try to infer from dataset name
        language = "python"

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i in range(start, end):
            try:
                # Access raw item to get all fields
                raw_item = dataset.dataset[i]
                
                # Get problem description and task_id
                # 'prompt' usually contains the description/signature
                problem_description = raw_item.get("prompt", "")
                task_id = raw_item.get("task_id", f"task_{i}")
                
                # Select Code Source based on perturbation type rules or argument
                # Rule: reverse_authority -> canonical, authority -> incorrect
                target_source = args.code_source
                if args.perturbation_type == "reverse_authority":
                    target_source = "canonical"
                elif args.perturbation_type == "authority":
                    target_source = "incorrect"

                validation_code = None
                if target_source == "canonical":
                    validation_code = raw_item.get("canonical_solution")
                elif target_source == "incorrect":
                    validation_code = raw_item.get("incorrect_solution")
                    if not validation_code:
                        print(f"Skipping item {i} ({task_id}): No incorrect_solution found.")
                        continue
                
                if not validation_code:
                    print(f"Skipping item {i} ({task_id}): No code found for source '{target_source}'.")
                    continue

                # Create Batch Request
                request_body = {
                    "custom_id": task_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model,
                        "messages": [
                            {"role": "system", "content": "You are a code perturbation assistant that adds misleading comments to code."},
                            {"role": "user", "content": create_perturbation_prompt(
                                validation_code, 
                                language, 
                                problem_description, 
                                args.perturbation_type
                            )}
                        ],
                        "temperature": 0
                    }
                }
                
                f_out.write(json.dumps(request_body) + "\n")
                requests_created += 1
                
            except Exception as e:
                print(f"Error processing item {i}: {e}")

    print(f"Batch file generated: {output_file}")
    print(f"Total items in range: {end - start}")
    print(f"Requests created: {requests_created}")
    
    return output_file

def submit_batch(batch_file, metadata=None):
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
        
        job_metadata = {
            "description": f"Perturbation batch job for {os.path.basename(batch_file)}"
        }
        if metadata:
            job_metadata.update(metadata)

        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=job_metadata
        )
        
        job_id = batch_job.id
        print(f"Batch job created. ID: {job_id}")
        
        # Save response
        output_dir = "batch_job_metadata"
        os.makedirs(output_dir, exist_ok=True)
        
        input_basename = os.path.basename(batch_file)
        if input_basename.endswith('.jsonl'):
            input_basename = input_basename[:-6]
        
        output_file = os.path.join(output_dir, f"{input_basename}_job.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if hasattr(batch_job, 'model_dump_json'):
                f.write(batch_job.model_dump_json(indent=2))
            else:
                f.write(json.dumps(batch_job.__dict__, indent=2, default=str))
        
        print(f"Job details saved to: {output_file}")
        print(f"\nTo check status and process later, use process_perturbation_batch.py")
        
    except Exception as e:
        print(f"Error submitting batch: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate OpenAI Batch API requests for code perturbation.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., 'leetcode', 'humaneval_py')")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model for perturbation (default: 'gpt-4o-mini')")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--output_dir", type=str, default="batch_jobs_perturb", help="Directory to save batch files")
    parser.add_argument("--dont_submit", action="store_true", help="Don't submit batch to OpenAI (only generate the file)")
    parser.add_argument("--perturbation_type", type=str, default="default", 
                        choices=["default", "misleading_task", "reverse_authority", "authority"],
                        help="Type of perturbation prompt to use")
    parser.add_argument("--code_source", type=str, default="canonical",
                        choices=["canonical", "incorrect"],
                        help="Source of code to perturb (canonical solution or incorrect solution)")

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
    
    # Generate batch
    batch_file = generate_perturbation_batch(args, dataset)
    
    # Submit batch if not disabled
    if not args.dont_submit:
        print("\nSubmitting batch to OpenAI...")
        
        # Determine effective source for metadata
        effective_source = args.code_source
        if args.perturbation_type == "reverse_authority":
            effective_source = "canonical"
        elif args.perturbation_type == "authority":
            effective_source = "incorrect"

        # Prepare metadata to save with the job
        metadata = {
            "dataset": args.dataset,
            "perturbation_type": args.perturbation_type,
            "code_source": effective_source
        }
        
        submit_batch(batch_file, metadata)
    else:
        print("\nBatch file created but not submitted (--dont_submit flag set)")

if __name__ == "__main__":
    main()
