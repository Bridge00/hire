import argparse
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Submit OpenAI Batch API job.")
    parser.add_argument("--batch_input_file", type=str, required=True, help="Path to the .jsonl batch input file")
    
    args = parser.parse_args()

    if not os.path.exists(args.batch_input_file):
        print(f"Error: Input file {args.batch_input_file} does not exist.")
        sys.exit(1)

    try:
        client = OpenAI()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY is set in your .env file or environment variables.")
        sys.exit(1)

    print(f"Uploading file: {args.batch_input_file}")
    
    try:
        batch_input_file = client.files.create(
            file=open(args.batch_input_file, "rb"),
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
              "description": f"Batch job for {os.path.basename(args.batch_input_file)}"
            }
        )
        
        job_id = batch_job.id
        print(f"Batch job created. ID: {job_id}")
        
        # Save response
        output_dir = "batch_response"
        os.makedirs(output_dir, exist_ok=True)
        
        input_basename = os.path.basename(args.batch_input_file)
        # Remove extension for cleaner name
        if input_basename.endswith('.jsonl'):
            input_basename = input_basename[:-6]
            
        output_file = os.path.join(output_dir, f"{input_basename}_job.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # batch_job is a Pydantic model in recent SDKs, or object. 
            # Using model_dump_json() if available, or fallback to default serialization or dict
            if hasattr(batch_job, 'model_dump_json'):
                 f.write(batch_job.model_dump_json(indent=2))
            elif hasattr(batch_job, 'to_json'):
                 f.write(json.dumps(batch_job.to_json(), indent=2))
            else:
                 # Fallback for older versions or if it's a dict/simple object
                 # Assuming it might be a pydantic model but checking dict first
                 try:
                     f.write(json.dumps(batch_job.__dict__, indent=2, default=str))
                 except:
                     f.write(str(batch_job))

        print(f"Job details saved to: {output_file}")
        
    except Exception as e:
        print(f"Error submitting batch: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
