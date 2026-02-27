import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_data_file_path(dataset_name):
    """Get the absolute path to the dataset JSONL file."""
    data_dir = os.getenv("DATA_DIR", "data")
    
    if dataset_name.lower() == "leetcode":
        filename = 'leetcode-hard.jsonl'
    elif dataset_name.lower() == "humaneval":
        filename = 'HumanEval.jsonl'
    elif "evoeval" in dataset_name.lower():
        filename = f'EvoEval_{dataset_name[dataset_name.find("_")+1:]}.jsonl'
    elif "core_eval" in dataset_name.lower():
        filename = f'core_eval.jsonl'
    elif dataset_name.lower().startswith("apps_"):
        filename = dataset_name.lower() + '.jsonl'
    else:
        filename = dataset_name.lower() + '.jsonl'
    
    return os.path.join(data_dir, filename)

def normalize_task_id(task_id):
    """Normalize task_id to match custom_id format used in batch."""
    if "/" in task_id:
        return '_'.join(task_id.split('/'))
    return task_id

def process_perturbation_results(results_file, dataset_name, key_name):
    """Process batch perturbation results and save back to dataset file."""
    data_path = get_data_file_path(dataset_name)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset file not found at {data_path}")
        return
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        return

    print(f"Processing results from {results_file} for dataset {dataset_name} ({data_path})")
    print(f"Injecting perturbations into key: '{key_name}'")
    
    # 1. Load results into a mapping: normalized_task_id -> perturbed_solution
    perturbations = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                
                if result.get("error"):
                    print(f"Warning: Request failed for {custom_id}. Error: {result['error']}")
                    continue
                
                response = result.get("response")
                if response and response.get("body"):
                    choices = response["body"].get("choices")
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        
                        # Remove markdown code blocks if present
                        if content.startswith("```"):
                            lines = content.split('\n')
                            # Remove first line (```language) and last line (```)
                            content = '\n'.join(lines[1:-1])
                        
                        perturbations[custom_id] = content
            except Exception as e:
                print(f"Error parsing result line: {e}")

    if not perturbations:
        print("No successful perturbations found in results.")
        return

    print(f"Loaded {len(perturbations)} perturbations.")

    # 2. Load the original dataset, update it, and save it
    updated_dataset = []
    update_count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                task_id = item.get("task_id")
                norm_task_id = normalize_task_id(task_id)
                
                # Use raw task_id for lookup to match batch results custom_id
                # Only normalize if necessary (but generation uses raw task_id)
                if task_id in perturbations:
                    item[key_name] = perturbations[task_id]
                    update_count += 1
                elif normalize_task_id(task_id) in perturbations:
                    # Fallback to normalized ID just in case
                    item[key_name] = perturbations[normalize_task_id(task_id)]
                    update_count += 1
                
                updated_dataset.append(item)
            except Exception as e:
                print(f"Error processing dataset line: {e}")
                updated_dataset.append(line.strip()) # Keep original line if failed to parse

    # 3. Save back to the original file (overwrite)
    if update_count > 0:
        with open(data_path, 'w', encoding='utf-8') as f:
            for item in updated_dataset:
                f.write(json.dumps(item) + '\n')
        print(f"Successfully updated {update_count} entries in {data_path}")
    else:
        print(f"No entries were updated in {data_path}")

from openai import OpenAI

def download_batch_results(job_file):
    """Check batch status and download results if completed."""
    if not os.path.exists(job_file):
        print(f"Error: Job file {job_file} does not exist.")
        sys.exit(1)

    try:
        with open(job_file, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
            batch_id = job_data.get("id")
            if not batch_id:
                print("Error: Could not find 'id' in job file.")
                sys.exit(1)
    except Exception as e:
        print(f"Error reading job file: {e}")
        sys.exit(1)

    try:
        client = OpenAI()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file or environment variables.")
        sys.exit(1)

    print(f"Checking status for Batch ID: {batch_id}")
    
    try:
        batch = client.batches.retrieve(batch_id)
        print(f"Status: {batch.status}")
        
        if batch.status == "completed":
            output_file_id = batch.output_file_id
            if not output_file_id:
                print("Batch is completed but has no output_file_id.")
                sys.exit(1)
                
            print(f"Downloading results from File ID: {output_file_id}")
            content = client.files.content(output_file_id).text
            
            # Determine output path
            job_basename = os.path.basename(job_file)
            if job_basename.endswith("_job.json"):
                base_name = job_basename[:-9]
            elif job_basename.endswith(".json"):
                base_name = job_basename[:-5]
            else:
                base_name = job_basename
            
            output_dir = "batch_results"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_results.jsonl")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"Results saved to: {output_path}")
            return output_path
            
        elif batch.status == "failed":
            print(f"Batch failed. Error: {batch.errors}")
            sys.exit(1)
        else:
            print("Batch is not yet completed. Please try again later.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error retrieving batch info: {e}")
        sys.exit(1)

def parse_filename_metadata(filename):
    """
    Parse metadata from the batch filename.
    Format: {dataset}_perturbation_{model}[_{source}]_[type]_{start}_{end}.jsonl
    """
    basename = os.path.basename(filename)
    if basename.endswith("_job.json"):
        basename = basename[:-9]
    elif basename.endswith("_results.jsonl"):
        basename = basename[:-14]
    elif basename.endswith(".jsonl"):
        basename = basename[:-6]
        
    parts = basename.split('_')
    
    # Heuristics
    dataset = None
    perturbation_type = None
    code_source = "canonical" # default
    
    # Try to find 'perturbation'
    try:
        p_idx = parts.index('perturbation')
        dataset = "_".join(parts[:p_idx])
        
        # After perturbation comes the model, then potentially source, type, start, end
        # model is usually gpt-4o-mini (dashes)
        # parts after p_idx: ['gpt-4o-mini', 'misleading', 'task', '0', '1'] (source canonical absent)
        # OR ['gpt-4o-mini', 'incorrect', 'authority', '0', '1']
        
        # Find numeric start/end at the end
        if parts[-1].isdigit() and parts[-2].isdigit():
            suffix_parts = parts[p_idx+1:-2]
        else:
            suffix_parts = parts[p_idx+1:]
            
        # Model is usually first item in suffix
        # Remove model
        if len(suffix_parts) > 0 and ("gpt" in suffix_parts[0] or "claude" in suffix_parts[0]):
            suffix_parts = suffix_parts[1:]
            
        remaining = "_".join(suffix_parts)
        
        # Detect source
        if "incorrect" in remaining:
            code_source = "incorrect_solution"
            remaining = remaining.replace("incorrect_", "").replace("_incorrect", "")
        elif "canonical" in remaining:
            code_source = "canonical_solution"
            remaining = remaining.replace("canonical_", "").replace("_canonical", "")
            
        # Remaining should be perturbation type
        # e.g. "misleading_task", "authority", "reverse_authority"
        if remaining:
            perturbation_type = remaining
            
    except ValueError:
        pass
        
    return dataset, perturbation_type, code_source

def main():
    parser = argparse.ArgumentParser(description="Process OpenAI Batch API results for code perturbation.")
    parser.add_argument("--results_file", type=str, help="Path to the batch results .jsonl file")
    parser.add_argument("--job_file", type=str, help="Path to the batch job .json file (will download results automatically)")
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g., 'leetcode', 'humaneval_py')")
    parser.add_argument("--perturbation_type", type=str, 
                        help="Type of perturbation used (part of key name)")
    parser.add_argument("--code_source", type=str,
                        help="Source code used (part of key name)")

    args = parser.parse_args()
    
    dataset = args.dataset
    perturbation_type = args.perturbation_type
    code_source = args.code_source
    
    results_file = None
    job_metadata = {}
    
    if args.job_file:
        print(f"Job file provided. Attempting to download results...")
        
        # Load metadata from job file first if possible (before download might fail)
        if os.path.exists(args.job_file):
            try:
                with open(args.job_file, 'r', encoding='utf-8') as f:
                    job_data = json.load(f)
                    if "metadata" in job_data and job_data["metadata"]:
                        job_metadata = job_data["metadata"]
                        print(f"Found metadata in job file: {job_metadata}")
            except Exception as e:
                print(f"Warning: Could not read metadata from job file: {e}")

        results_file = download_batch_results(args.job_file)
    elif args.results_file:
        results_file = args.results_file
    else:
        print("Error: Must provide either --results_file or --job_file")
        sys.exit(1)
        
    # Resolution Logic
    
    # 1. Dataset
    if not dataset:
        dataset = job_metadata.get("dataset")
    if not dataset:
        # Try filename
        d, _, _ = parse_filename_metadata(results_file)
        if d: dataset = d
    
    if not dataset:
        print("Error: Could not determine dataset. Please provide --dataset argument.")
        sys.exit(1)
        
    # 2. Code Source
    if not code_source:
        code_source = job_metadata.get("code_source")
        
    # Check filename for effective source override (to fix bug where metadata says canonical but file is incorrect)
    _, _, filename_source = parse_filename_metadata(results_file)
    
    if code_source == "canonical" and filename_source == "incorrect_solution":
        print("Warning: Metadata claims 'canonical' but filename indicates 'incorrect_solution'. Trusting filename.")
        code_source = "incorrect_solution"

    if not code_source:
        code_source = filename_source

    if not code_source:
        code_source = "canonical_solution" # Final fallback

    # 3. Perturbation Type
    if not perturbation_type:
        perturbation_type = job_metadata.get("perturbation_type")
    if not perturbation_type:
        _, p, _ = parse_filename_metadata(results_file)
        perturbation_type = p
        
    if not perturbation_type:
        print("Error: Could not determine perturbation_type. Please provide --perturbation_type argument (e.g., 'misleading_task').")
        sys.exit(1)
        
    print(f"Resolved Parameters: Dataset='{dataset}', Type='{perturbation_type}', Source='{code_source}'")
    
    key_name = f"{perturbation_type}_{code_source}"
    
    process_perturbation_results(results_file, dataset, key_name)

if __name__ == "__main__":
    main()
