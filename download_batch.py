import argparse
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
try:
    from together import Together
except ImportError:
    Together = None

from process_batch_results import process_and_cache_results

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Download Batch API results (OpenAI or Together AI).")
    parser.add_argument("--job_file", type=str, required=True, help="Path to the .json job file created by generate_batch.py")
    
    args = parser.parse_args()

    if not os.path.exists(args.job_file):
        print(f"Error: Job file {args.job_file} does not exist.")
        sys.exit(1)

    try:
        with open(args.job_file, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
            batch_id = job_data.get("id")
            if not batch_id:
                print("Error: Could not find 'id' in job file.")
                sys.exit(1)
    except Exception as e:
        print(f"Error reading job file: {e}")
        sys.exit(1)

    # Detect API type
    is_together = False
    if "project_id" in job_data or ("object" not in job_data and not batch_id.startswith("batch_")):
        is_together = True
        if Together is None:
            print("Error: together library not found but Together job detected.")
            sys.exit(1)

    try:
        if is_together:
            print(f"Detected Together AI job. Initializing Together client...")
            client = Together()
        else:
            print(f"Detected OpenAI job. Initializing OpenAI client...")
            client = OpenAI()
    except Exception as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)

    print(f"Checking status for Batch ID: {batch_id}")
    
    try:
        if is_together:
            batch = client.batches.retrieve(batch_id)
            # Together AI status is uppercase
            status = batch.status.upper()
            print(f"Status: {status}")
            
            if status == "COMPLETED":
                output_file_id = batch.output_file_id
                if not output_file_id:
                    print("Batch completed but output_file_id is missing.")
                    sys.exit(1)
                
                print(f"Downloading results from File ID: {output_file_id}")
                # Use requests to avoid the SDK's Windows lock file bugs
                import requests
                api_key = os.environ.get("TOGETHER_API_KEY")
                url = f"https://api.together.xyz/v1/files/{output_file_id}/content"
                headers = {"Authorization": f"Bearer {api_key}"}
                
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                content = response.text
                
            elif status == "FAILED":
                print(f"Batch failed. Error: {getattr(batch, 'error', 'Unknown error')}")
                sys.exit(1)
            elif status in ["CANCELLED", "CANCELING"]:
                print(f"Batch was {status.lower()}.")
                sys.exit(1)
            else:
                print(f"Batch is still {status.lower()}. Please try again later.")
                sys.exit(0)
        else:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            print(f"Status: {status}")
            
            if status == "completed":
                if batch.request_counts and batch.request_counts.failed > 0:
                    print(f"Warning: Batch had {batch.request_counts.failed} failures (out of {batch.request_counts.total}).")
                    if batch.error_file_id:
                        print(f"Downloading error details from File ID: {batch.error_file_id}")
                        try:
                            error_content = client.files.content(batch.error_file_id).text
                            print("Error details (first 1000 chars):")
                            print(error_content[:1000] + ("..." if len(error_content) > 1000 else ""))
                        except Exception as e:
                            print(f"Could not download error file: {e}")
                
                output_file_id = batch.output_file_id
                if not output_file_id:
                    print("Batch completed but output_file_id is missing.")
                    sys.exit(1)
                    
                print(f"Downloading results from File ID: {output_file_id}")
                content = client.files.content(output_file_id).text
            elif status == "failed":
                print(f"Batch failed. Error: {batch.errors}")
                sys.exit(1)
            else:
                print("Batch is not yet completed. Please try again later.")
                sys.exit(0)

        # Common download and processing logic
        # Get basename of job file
        job_basename = os.path.basename(args.job_file)
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
        
        # Automatically process and cache the results
        print("Automatically processing and caching results...")
        cache_dir = os.environ.get("CACHE_DIR", ".cache")
        
        # Parse metadata from filename
        from process_batch_results import parse_filename_metadata
        dataset, code_gen_model, eval_model, eval_prompt_type = parse_filename_metadata(output_path)
        
        if dataset and code_gen_model:
            process_and_cache_results(output_path, dataset, code_gen_model, eval_model, eval_prompt_type, cache_dir)
        else:
            print("Warning: Could not auto-detect metadata from filename. Skipping automatic cache processing.")
            print("Please run process_batch_results.py manually with appropriate arguments.")
            
    except Exception as e:
        print(f"Error retrieving batch info or downloading results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
