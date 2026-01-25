#!/bin/bash

# Default hyperparameters
DATASETS=("leetcode" "humaneval_py")
EVAL_METHODS=("vanilla" "cj_analysis" "cj_summary" "hire_decomposer" "hire_plan_checker")
EVAL_MODELS=("gpt-4o" "gpt-4o-mini")
CODE_GEN_MODEL="gpt-4o-mini"
MODE=""
DONT_SUBMIT=""
PYTHON_CMD=""

# Help message
usage() {
    echo "Usage: $0 --mode <generate|process> [options]"
    echo ""
    echo "Options:"
    echo "  --mode <generate|process>  Required: Mode of operation"
    echo "  --datasets \"ds1 ds2\"       Space-separated list of datasets (default: ${DATASETS[*]})"
    echo "  --eval_methods \"m1 m2\"     Space-separated list of eval methods (default: ${EVAL_METHODS[*]})"
    echo "  --eval_models \"mod1 mod2\"  Space-separated list of eval models (default: ${EVAL_MODELS[*]})"
    echo "  --python <path>            Path to python executable (default: auto-detect)"
    echo "  --dont_submit              Generate batch files but don't submit (only for generate mode)"
    echo "  --force                    Force generation even if item is already in cache"
    echo "  --help                     Show this help message"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --datasets) IFS=' ' read -r -a DATASETS <<< "$2"; shift ;;
        --eval_methods) IFS=' ' read -r -a EVAL_METHODS <<< "$2"; shift ;;
        --eval_models) IFS=' ' read -r -a EVAL_MODELS <<< "$2"; shift ;;
        --code_gen_model) CODE_GEN_MODEL="$2"; shift ;;
        --python) PYTHON_CMD="$2"; shift ;;
        --dont_submit) DONT_SUBMIT="--dont_submit" ;;
        --force) FORCE="--force" ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required."
    usage
fi

if [[ "$MODE" != "generate" && "$MODE" != "process" ]]; then
    echo "Error: --mode must be 'generate' or 'process'."
    usage
fi

echo "--- Hyperparameter Sweep ---"
echo "Mode: $MODE"
echo "Datasets: ${DATASETS[*]}"
echo "Eval Methods: ${EVAL_METHODS[*]}"
echo "Eval Models: ${EVAL_MODELS[*]}"
echo "Code Gen Model: $CODE_GEN_MODEL"
echo "----------------------------"

# Determine python command
if [[ -z "$PYTHON_CMD" ]]; then
    PYTHON_CMD="python"
    if ! command -v python &> /dev/null; then
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        else
            # Try to find osworld environment if on Windows-like system
            OSWORLD_PY="/c/Users/bhrij/miniconda3/envs/osworld/python.exe"
            if [[ -f "$OSWORLD_PY" ]]; then
                PYTHON_CMD="$OSWORLD_PY"
            else
                echo "Error: python or python3 not found, and osworld env not detected at $OSWORLD_PY."
                echo "Please specify python path with --python <path>"
                exit 1
            fi
        fi
    fi
fi

for dataset in "${DATASETS[@]}"; do
    for eval_method in "${EVAL_METHODS[@]}"; do
        for eval_model in "${EVAL_MODELS[@]}"; do
            
            if [[ "$MODE" == "generate" ]]; then
                echo "Generating batch for: Dataset=$dataset, EvalMethod=$eval_method, EvalModel=$eval_model"
                
                # Construct command
                echo "Running: \"$PYTHON_CMD\" generate_batch.py --mode eval --dataset $dataset --code_gen_model $CODE_GEN_MODEL --eval_model $eval_model --eval_prompt $eval_method $DONT_SUBMIT $FORCE"
                "$PYTHON_CMD" generate_batch.py --mode eval --dataset $dataset --code_gen_model $CODE_GEN_MODEL --eval_model $eval_model --eval_prompt $eval_method $DONT_SUBMIT $FORCE
                
            elif [[ "$MODE" == "process" ]]; then
                echo "Processing results for: Dataset=$dataset, EvalMethod=$eval_method, EvalModel=$eval_model"
                
                # For processing, we need to find the results file. 
                # The naming convention is: dataset_code_model_CODEMODEL_PROMPTTYPE_eval_eval_EVALMODEL_START_END_results.jsonl
                # We'll look for files matching this pattern in batch_results/
                
                # Note: This assumes the results have been downloaded and placed in batch_results/
                # with the expected naming convention.
                
                PATTERN="batch_results/${dataset}_code_model_${CODE_GEN_MODEL}_${eval_method}_eval_eval_${eval_model}_*_results.jsonl"
                
                # Check if any files match the pattern
                FILES=( $PATTERN )
                if [[ -e "${FILES[0]}" ]]; then
                    for file in "${FILES[@]}"; do
                        echo "Processing file: $file"
                        "$PYTHON_CMD" process_batch_results.py --results_file "$file"
                    done
                else
                    echo "No results files found for pattern: $PATTERN"
                fi
            fi
            
            echo "-----------------------------------"
        done
    done
done

echo "Sweep completed."
