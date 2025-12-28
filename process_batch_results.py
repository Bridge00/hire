import argparse
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Process OpenAI Batch API results and populate cache.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the results .jsonl file downloaded from OpenAI")

    args = parser.parse_args()
    
    # Determine cache directory

    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    processed_count = 0
    error_count = 0
    
    if not os.path.exists(args.results_file):
        print(f"Error: Input file {args.results_file} does not exist.")
        sys.exit(1)

    print(f"Processing results from: {args.results_file}")
    
    with open(args.results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                result = json.loads(line)
                
                # Verify structure
                custom_id = result.get("custom_id")
                
                if not custom_id:
                    print(f"Line {line_num}: Missing 'custom_id'. Skipping.")
                    error_count += 1
                    continue
                
                # Check for errors in the individual response
                if result.get("error"):
                    print(f"Line {line_num}: Request failed for {custom_id}. Error: {result['error']}")
                    error_count += 1
                    continue
                    
                response = result.get("response")
                if not response or not response.get("body"):
                    print(f"Line {line_num}: Missing response body for {custom_id}.")
                    error_count += 1
                    continue
                
                body = response["body"]
                choices = body.get("choices")
                if not choices or len(choices) == 0:
                     print(f"Line {line_num}: No choices in response for {custom_id}.")
                     error_count += 1
                     continue
                     
                content = choices[0].get("message", {}).get("content")
                if content is None:
                    print(f"Line {line_num}: No content in message for {custom_id}.")
                    error_count += 1
                    continue

                # Save to cache
                cache_path = os.path.join(cache_dir, f"{custom_id}.json")
                with open(cache_path, 'w', encoding='utf-8') as cache_f:
                    json.dump({"content": content}, cache_f)
                
                processed_count += 1
                
            except json.JSONDecodeError:
                print(f"Line {line_num}: Invalid JSON.")
                error_count += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error: {e}")
                error_count += 1

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Successfully processed and cached: {processed_count}")
    print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    main()
