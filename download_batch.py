import argparse
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Download OpenAI Batch API results.")
    parser.add_argument("--job_file", type=str, required=True, help="Path to the .json job file created by run_batch.py")
    
    args = parser.parse_args()

    if not os.path.exists(args.job_file):
        print(f"Error: Job file {args.job_file} does not exist.")
        sys.exit(1)

    try:
        with open(args.job_file, 'r', encoding='utf-8') as f:
            job_data = json.load(f)
            # Support both direct ID access and object-like structure if it was serialized differently
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
            # job_file is like batch_response/filename_job.json
            # target is batch_results/filename_results.jsonl
            
            # Get basename of job file
            job_basename = os.path.basename(args.job_file)
            # Expected format: {name}_job.json
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
            
        elif batch.status == "failed":
            print(f"Batch failed. Error: {batch.errors}")
        else:
            print("Batch is not yet completed. Please try again later.")
            
    except Exception as e:
        print(f"Error retrieving batch info: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
