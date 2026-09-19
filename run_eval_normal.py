import argparse
import os
import sys
import copy
from dotenv import load_dotenv

from data.all_code_benchmarks import CodeData
from src.eval_class import run_direct_eval

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Evaluate code snippets using Normal API and handle dependencies automatically.")
    parser.add_argument("--dataset", type=str, required=True, nargs='+', help="Name of the dataset(s) (e.g., 'leetcode')")
    parser.add_argument("--code_gen_model", type=str, nargs='+', required=False, help="Model(s) used to generate the code (e.g., 'gpt-4o-mini')")
    parser.add_argument("--eval_source", type=str, nargs='+', required=False, help="Source key(s) instead of generated code (e.g. 'incorrect_solution')")
    parser.add_argument("--eval_model", type=str, nargs='+', required=True, help="Model(s) to use for evaluation (e.g., 'llama31-8b')")
    parser.add_argument("--explainer_model", type=str, default=None, help="Explainer model name if evaluation depends on explanations outputted by another model.")
    parser.add_argument("--style_transfer_model", type=str, default=None, help="Model for explanation style-transfer (defaults to explainer_model)")
    parser.add_argument("--eval_prompt", type=str, nargs='+', required=True, help="Evaluation prompt(s) to use")
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "dialogue", "dialogue_n_questions", "dialogue_n_questions_rc", "dialogue_n_questions_rc_exact"], help="Running mode (defaults to 'eval')")
    parser.add_argument("--explainer_style", type=str, default="original", choices=["original", "objective", "rationale_split"], help="Explainer answering style (defaults to 'original')")
    parser.add_argument("--compact_second_judge", action="store_true", help="Use HIRE_DIALOGUE_SECOND_JUDGE_USER_RC_COMPACT for failed rc_exact judge decisions")
    parser.add_argument("-requirements_only_judges", "--requirements_only_judges", dest="requirements_only_judges", action="store_true", help="Extract requirements first and expose judges only to that requirements list")
    parser.add_argument("--requirements_model", type=str, default=None, help="Model for requirements extraction (defaults to eval_model)")
    parser.add_argument("--force_requirements", action="store_true", help="Regenerate shared requirements even when they are already cached")
    parser.add_argument("-specifications_only_judges", "--specifications_only_judges", dest="specifications_only_judges", action="store_true", help="Extract specifications first and expose judges only to that specification list")
    parser.add_argument("--specifications_model", type=str, default=None, help="Model for specification extraction (defaults to eval_model)")
    parser.add_argument("--force_specifications", action="store_true", help="Regenerate shared specifications even when they are already cached")
    parser.add_argument("--specification_aware_dialogue", action="store_true", help="Use structured specifications to prevent dialogue judges from treating preconditions as validation obligations")
    parser.add_argument("--rcu_max_iterations", type=int, default=3, help="Maximum adaptive reconstruction/compare/update refinement iterations")
    parser.add_argument("--one_by_one_verifier", action="store_true", help="Verify conservative specifications independently and clarify only uncertain requirements")
    parser.add_argument("--dual_agent", type=str, default=None, help="Independent code evaluator to cross-check against the explanation, e.g. codejudge or behavior_comparison_no_rc")
    parser.add_argument("--appeal_requirements_extractor_model", type=str, default=None, help="Model for extracting code-free alleged violated requirements before the dual-agent appeal (defaults to eval_model)")
    parser.add_argument("--requirements_appeal_extractor_model", type=str, default=None, help="Model for extracting code-free alleged violated requirements in the single-judge requirements appeal (defaults to eval_model)")
    parser.add_argument("--combined_evaluator_order", choices=["explanation_then_code", "code_then_explanation"], default=None, help="Evaluate the style-transfer explanation and code together in the selected order")
    parser.add_argument("--conservative_specifications_model", type=str, default=None, help="Model for conservative specification extraction (defaults to eval_model)")
    parser.add_argument("--force_conservative_specifications", action="store_true", help="Regenerate shared conservative specifications")
    parser.add_argument("--start_problem", type=int, default=0, help="Start problem index")
    parser.add_argument("--end_problem", type=int, default=None, help="End problem index")
    parser.add_argument("--lambda_val", type=int, nargs='+', default=None, help="Lambda value(s) for the HIRE framework")
    parser.add_argument("--n", type=int, nargs='+', default=[3], help="Number of steps for decomposer")
    parser.add_argument("--tasks", type=str, nargs='+', default=None, help="Specific task IDs to process")
    parser.add_argument("--force", action="store_true", help="Force re-generation of the specified prompt (does NOT force dependencies)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for parallel processing")
    
    args = parser.parse_args()
    if args.requirements_only_judges and args.mode != "dialogue_n_questions":
        parser.error("--requirements_only_judges should be used with --mode dialogue_n_questions; RC modes are redundant")
    if args.specifications_only_judges and args.mode != "dialogue_n_questions":
        parser.error("--specifications_only_judges should be used with --mode dialogue_n_questions; RC modes are redundant")
    if args.specification_aware_dialogue and args.mode != "dialogue_n_questions":
        parser.error("--specification_aware_dialogue should be used with --mode dialogue_n_questions")
    if args.specification_aware_dialogue and (args.requirements_only_judges or args.specifications_only_judges):
        parser.error("--specification_aware_dialogue cannot be combined with requirements-only or specifications-only judges")
    if args.requirements_only_judges and args.specifications_only_judges:
        parser.error("--requirements_only_judges and --specifications_only_judges are mutually exclusive")
    if args.one_by_one_verifier and args.mode != "dialogue_n_questions":
        parser.error("--one_by_one_verifier should be used with --mode dialogue_n_questions")
    if args.one_by_one_verifier and (args.requirements_only_judges or args.specifications_only_judges):
        parser.error("--one_by_one_verifier cannot be combined with requirements-only or specifications-only judges")
    if args.dual_agent and args.mode != "dialogue_n_questions":
        parser.error("--dual_agent should be used with --mode dialogue_n_questions")
    if args.dual_agent and (args.one_by_one_verifier or args.requirements_only_judges or args.specifications_only_judges):
        parser.error("--dual_agent cannot be combined with other specialized judge pipelines")
    if args.combined_evaluator_order and args.mode != "dialogue_n_questions":
        parser.error("--combined_evaluator_order should be used with --mode dialogue_n_questions")
    if args.combined_evaluator_order and (args.dual_agent or args.one_by_one_verifier or args.requirements_only_judges or args.specifications_only_judges):
        parser.error("--combined_evaluator_order cannot be combined with other specialized judge pipelines")
    if not args.code_gen_model and not args.eval_source:
        print("Error: Must provide either --code_gen_model or --eval_source")
        sys.exit(1)

    datasets = args.dataset
    code_gen_models = args.code_gen_model if args.code_gen_model else [None]
    eval_sources = args.eval_source if args.eval_source else [None]
    eval_models = args.eval_model
    eval_prompts = args.eval_prompt
    lambda_vals = args.lambda_val if args.lambda_val else [None]
    n_vals = args.n
    cache_dir = os.environ.get("CACHE_DIR", ".cache")
    
    for d_name in datasets:
        print(f"\nProcessing dataset: {d_name}")
        dataset = CodeData(d_name)
        for cgm in code_gen_models:
            for es in eval_sources:
                if not cgm and not es: continue
                for em in eval_models:
                    for ep in eval_prompts:
                        for l_val in lambda_vals:
                            for n_val in n_vals:
                                current_args = copy.deepcopy(args)
                                current_args.dataset, current_args.code_gen_model, current_args.eval_source, current_args.eval_model, current_args.eval_prompt, current_args.lambda_val, current_args.n = d_name, cgm, es, em, ep, l_val, n_val
                                current_args.mode = args.mode
                                print(f"\n--- Sweep Config: Dataset={d_name}, Source={es or cgm}, EvalModel={em}, Prompt={ep}, Lambda={l_val}, N={n_val} ---")
                                run_direct_eval(current_args, dataset, cache_dir)

if __name__ == "__main__":
    main()
