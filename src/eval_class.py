import json
import os
import copy
import time
import concurrent.futures
import re
from utils.llm import clean_code, is_together_model, _sanitize_filename, count_tokens
import utils.prompts as up

# System Prompts
EVAL_SYS = "You are a smart language model that evaluates code snippets. You do not solve problems or propose new code snippets, only evaluate existing solutions critically and give very concise critiques."
EXPLAINER_SYS = "You are a helpful AI assistant that explains code snippets in clear, accurate natural language. Your goal is to be descriptive and objective."
HIRE_PSEUDO_SYS = "You are a critical code evaluator. Your task is to determine if a given pseudocode accurately reflects the logic required by a problem description. You must be rigorous and identify any missing logic or incorrect assumptions in the pseudocode relative to the task requirements."
HIRE_EXPLAINER_SYS = "You are a critical code evaluator. Your task is to determine if a natural language explanation of a code snippet accurately and completely covers the requirements of a problem description. You must ensure the explanation is logically sound and aligns perfectly with the task goals."
HIRE_EXPLAINER_SYS_FAITHFUL = "You are a rigorous code evaluator. Your task is to determine if a natural language explanation of a code snippet accurately and completely covers the requirements of a problem description. **WATCH FOR HALLUCINATIONS**: Do not be fooled by explanations that claim success in walkthroughs while describing flawed logic in the algorithm section. Ensure the described logic AND the walkthroughs are both correct and consistent with the problem."
FEEDBACK_SYS = "You are a critical code reviewer. Your goal is to identify discrepancies between code and its explanation by looking at an implementation reconstructed from that explanation. Provide actionable feedback to improve the explanation."
UPDATE_SYS = "You are an expert technical writer. Your goal is to update a code explanation based on feedback and the original source code to ensure perfect accuracy and clarity."

MODEL_MAPPING = {
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen25-72b": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "llama31-8b": "Bridge00/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-62c81e95",
    "gpt-5.4-thinking": "gpt-5.4",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-sonnet-4.6": "claude-sonnet-4-6"
}

def resolve_model(model_name):
    return MODEL_MAPPING.get(model_name, model_name)

def augment_code_with_comments(original_code, steps):
    augmented = ""
    cursor = 0
    for step in steps:
        expl = step.get("explanation", "").replace("\n", " ")
        seg = step.get("code_segment", "")
        if not seg: continue
        idx = original_code.find(seg, cursor)
        match_len = len(seg)
        if idx == -1:
            lines = seg.strip().split('\n')
            if lines:
                first_line = lines[0].strip()
                candidate = original_code.find(first_line, cursor)
                if candidate != -1:
                    idx = candidate
                    match_len = len(first_line)
        if idx != -1:
            augmented += original_code[cursor:idx]
            if augmented and not augmented.endswith('\n'): augmented += "\n"
            augmented += f"# SUMMARIZER AGENT : {expl}\n"
            augmented += original_code[idx : idx + match_len]
            cursor = idx + match_len
        else:
            if augmented and not augmented.endswith('\n'): augmented += "\n"
            augmented += f"# SUMMARIZER AGENT : {expl} (Location Approx)\n"
    augmented += original_code[cursor:]
    return augmented

class EvaluatorRunner:
    def __init__(self, args, dataset, cache_dir):
        self.args = args
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.model_or_source = (getattr(args, 'eval_source', None) or 
                                getattr(args, 'code_gen_model', None) or 
                                getattr(args, 'explainer_model', None) or 
                                args.eval_model)
        self.model_name = f"{args.explainer_model}_{args.eval_model}" if hasattr(args, 'explainer_model') and args.explainer_model else args.eval_model

    def _resolve_programming_language(self, row):
        if "language" in row:
            raw_lang = row["language"].lower()
            if "python" in raw_lang: return "python"
            elif "cpp" in raw_lang or "c++" in raw_lang: return "cpp"
            elif "java" in raw_lang: return "java"
            elif "js" in raw_lang or "javascript" in raw_lang: return "javascript"
            elif "go" in raw_lang: return "go"
        
        if self.args.dataset in ["leetcode", "humaneval_py", "debugbench_hard"]: return "python"
        elif self.args.dataset == "humaneval_js": return "javascript"
        elif self.args.dataset == "humaneval_java": return "java"
        elif self.args.dataset == "humaneval_cpp": return "c++"
        elif self.args.dataset == "humaneval_go": return "go"
        return "python"

    def _do_llm_call(self, sys_prompt, user_prompt, task_id, eval_model, eval_prompt_type, force=False, model_name_override=None, cache_path_override=None, cache_model_folder=None):
        cache_model_folder = cache_model_folder or eval_model
        cache_path = cache_path_override or os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(cache_model_folder), _sanitize_filename(eval_prompt_type), f"{_sanitize_filename(task_id)}.json")
        if not force and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                if isinstance(cache_data, dict):
                    content = cache_data.get("content", "")
                    if "input_tokens" in cache_data:
                        print(f"[CACHED] [{eval_model}/{eval_prompt_type}] Latency: {cache_data.get('latency', 0.0):.2f}s | Input Tokens: {cache_data.get('input_tokens', 0)} | Output Tokens: {cache_data.get('output_tokens', 0)}")
                    else:
                        input_tokens = count_tokens(sys_prompt, eval_model) + count_tokens(user_prompt, eval_model)
                        output_tokens = count_tokens(content, eval_model)
                        print(f"[CACHED-ESTIMATED] [{eval_model}/{eval_prompt_type}] Prompt Tokens: {input_tokens} | Output Tokens: {output_tokens}")
                    return content
                return str(cache_data)
        
        target_model = model_name_override if model_name_override else self.args.eval_model
        api_model_name = resolve_model(target_model)
        
        start_time = time.time()
        response_obj = None

        extra_kwargs = {}
        if is_together_model(api_model_name):
            from together import Together
            client = Together()
            response = client.chat.completions.create(
                model=api_model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                **extra_kwargs
            )
            response_obj = response
            content = response.choices[0].message.content
        elif api_model_name.startswith("gemini-"):
            from openai import OpenAI
            gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for Gemini models.")
            client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            if "gemini-3" in api_model_name or "gemini-3.5" in api_model_name or "thinking" in api_model_name:
                extra_kwargs["reasoning_effort"] = "high"
            response = client.chat.completions.create(
                model=api_model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                **extra_kwargs
            )
            response_obj = response
            content = response.choices[0].message.content
        elif api_model_name.startswith("claude-"):
            from anthropic import Anthropic
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required for Claude models.")
            client = Anthropic(api_key=anthropic_api_key)
            if "opus-4.6" in api_model_name or "sonnet-4.6" in api_model_name or "3-7-sonnet" in api_model_name or "thinking" in api_model_name:
                extra_kwargs["thinking"] = {"type": "adaptive"}
                extra_kwargs["max_tokens"] = 16384
            else:
                extra_kwargs["max_tokens"] = 4096
            response = client.messages.create(
                model=api_model_name,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                **extra_kwargs
            )
            response_obj = response
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
        else:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=api_model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                **extra_kwargs
            )
            response_obj = response
            content = response.choices[0].message.content
            
        latency = time.time() - start_time
        
        # Extract prompt/completion tokens
        input_tokens = 0
        output_tokens = 0
        try:
            if response_obj and hasattr(response_obj, "usage") and response_obj.usage:
                if hasattr(response_obj.usage, "prompt_tokens"):
                    input_tokens = response_obj.usage.prompt_tokens
                    output_tokens = response_obj.usage.completion_tokens
                elif hasattr(response_obj.usage, "input_tokens"):
                    input_tokens = response_obj.usage.input_tokens
                    output_tokens = response_obj.usage.output_tokens
        except Exception:
            pass

        if not input_tokens:
            input_tokens = count_tokens(sys_prompt, api_model_name) + count_tokens(user_prompt, api_model_name)
        if not output_tokens:
            output_tokens = count_tokens(content, api_model_name)

        print(f"[API CALL] [{api_model_name}/{eval_prompt_type}] Latency: {latency:.2f}s | Input Tokens: {input_tokens} | Output Tokens: {output_tokens}")
        
        cache_data = {
            "content": content,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
        return content

    def run_direct_judge_evidence_appeal(self, task_id, problem_prompt, cleaned_code,
                                         prompt_type, force=False):
        """Audit negative direct-code verdicts against specifications and source evidence."""
        from utils.mcts_judge import parse_json

        args = self.args
        base_prompt = "vanilla" if prompt_type.startswith("vanilla_") else "cj_summary"
        cache_path = os.path.join(self.cache_dir, args.dataset, _sanitize_filename(self.model_or_source),
                                  _sanitize_filename(args.eval_model), _sanitize_filename(prompt_type),
                                  f"{_sanitize_filename(task_id)}.json")
        if not force and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f).get("content", "")

        initial_response = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, base_prompt, force=False)
        initial = parse_json(initial_response, default=None)
        if not isinstance(initial, dict):
            text_verdict = str(initial_response).strip().lower()
            initial = {
                "correct": text_verdict in {"yes", "true", "correct"},
                "reasoning": str(initial_response),
            }
        initial_value = initial.get("correct", False)
        initial_correct = initial_value.strip().lower() in ("true", "yes", "1") if isinstance(initial_value, str) else bool(initial_value)
        output = {"correct": initial_correct, "verdict": "Yes" if initial_correct else "No",
                  "direct_judge": base_prompt, "first_judgment": initial,
                  "specification": None, "mapped_violations": [], "requirements_appeal": None}
        if not initial_correct:
            specification_model = getattr(args, "specifications_model", None) or args.eval_model
            specification_path = os.path.join(self.cache_dir, args.dataset, "specifications",
                                              _sanitize_filename(specification_model), f"{_sanitize_filename(task_id)}.json")
            specification_response = self._do_llm_call(
                "You extract authoritative structured specifications without adding unstated obligations.",
                up.HIRE_SPECIFICATION_EXTRACTOR_USER.format(PROBLEM=problem_prompt), task_id, specification_model,
                "specification_extractor", force=getattr(args, "force_specifications", False),
                model_name_override=specification_model, cache_path_override=specification_path,
            )
            specification = parse_json(specification_response, default={})
            specification_text = json.dumps(specification, indent=2)
            eligible = []
            if isinstance(specification, dict):
                eligible = specification.get("functional_obligations", []) + specification.get("output_requirements", [])
            valid_ids = {str(item.get("id")) for item in eligible if isinstance(item, dict) and item.get("id")}
            output["specification"] = specification
            mapper_response = self._do_llm_call(
                "Faithfully map stated direct-code failure reasons to an authoritative specification without adding failures.",
                up.STYLE_TRANSFER_NEGATIVE_JUDGMENT_MAPPER.format(
                    PROBLEM=problem_prompt, SPECIFICATION=specification_text,
                    JUDGMENT=json.dumps(initial, indent=2),
                ), task_id, args.eval_model, f"{prompt_type}/negative_judgment_mapper", force=force,
                model_name_override=args.eval_model, cache_model_folder=args.eval_model,
            )
            mapped = parse_json(mapper_response, default={})
            mappings = mapped.get("failure_mappings", []) if isinstance(mapped, dict) else []
            violations = [v for v in mappings if isinstance(v, dict) and str(v.get("requirement_id", "")) in valid_ids]
            output["mapped_violations"] = violations
            mapped_judgment = {"correct": False, "reasoning": initial.get("reasoning", "") if isinstance(initial, dict) else initial_response,
                               "violations": mappings}
            appeal_response = self._do_llm_call(
                "Audit mapped direct-code findings against original problem text, examples, and specifications.",
                up.STYLE_TRANSFER_SPECIFICATION_EVIDENCE_APPEAL.format(
                    PROBLEM=problem_prompt, SPECIFICATION=specification_text,
                    JUDGE_DECISION=json.dumps(mapped_judgment, indent=2),
                ), task_id, args.eval_model, f"{prompt_type}/requirements_mapping_appeal", force=force,
                model_name_override=args.eval_model, cache_model_folder=args.eval_model,
            )
            appeal = parse_json(appeal_response, default={})
            output["requirements_appeal"] = appeal
            appeal_violations = [v for v in appeal.get("violations", []) if isinstance(v, dict) and str(v.get("requirement_id", "")) in valid_ids] if isinstance(appeal, dict) else []
            complete = all(isinstance(v.get("text_evidence"), list) and isinstance(v.get("example_evidence"), list)
                           and bool(str(v.get("strongest_joint_interpretation", "")).strip())
                           and str(v.get("classification", "")).lower() in {"explicitly_supported", "logically_implied", "ambiguous", "unsupported"}
                           and isinstance(v.get("does_observed_behavior_violate_it"), bool) for v in appeal_violations)
            supported = any(str(v.get("classification", "")).lower() in {"explicitly_supported", "logically_implied"}
                            and v.get("does_observed_behavior_violate_it") is True for v in appeal_violations)
            if violations and len(appeal_violations) == len(violations) and complete and not supported:
                output["correct"], output["verdict"] = True, "Yes"
        trace = json.dumps(output, indent=2)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def run_rcu_refine(self, task_id, problem_prompt, cleaned_code, prompt_type, force=False):
        """Static, discrepancy-driven reconstruction/compare/update refinement."""
        from utils.mcts_judge import parse_json

        def normalize_explanation(value):
            """Turn a model's accidental structured patch into faithful plain text."""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, dict):
                parts = []
                for key, item in value.items():
                    rendered = normalize_explanation(item)
                    if rendered:
                        parts.append(f"{str(key).replace('_', ' ').capitalize()}:\n{rendered}")
                return "\n\n".join(parts)
            if isinstance(value, (list, tuple)):
                parts = [normalize_explanation(item) for item in value]
                return "\n\n".join(f"- {item}" for item in parts if item)
            if value is None:
                return ""
            return str(value).strip()

        args = self.args
        source_model = getattr(args, "explainer_model", None) or args.eval_model
        cache_pt = prompt_type + (f"_lambda_L{args.lambda_val}" if args.lambda_val else "")
        path = os.path.join(self.cache_dir, args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(source_model), _sanitize_filename(cache_pt), f"{_sanitize_filename(task_id)}.json")
        if not force and os.path.exists(path):
            with open(path, encoding="utf-8") as f: return json.load(f).get("content", "")
        source = f"hire_explainer_obj_lambda_L{args.lambda_val}" if args.lambda_val else "hire_explainer_obj"
        explanation = normalize_explanation(
            self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=False)
        )
        reconstruction_language = "C++" if args.dataset == "humaneval_cpp" else "Python"
        original_facts_raw = self._do_llm_call(EVAL_SYS, up.RCU_FACTS.format(CODE=cleaned_code), task_id, args.eval_model, f"{cache_pt}/original_facts", force=force, model_name_override=args.eval_model, cache_model_folder=source_model)
        original_facts = parse_json(original_facts_raw, default={})
        iterations=[]
        for i in range(getattr(args, "rcu_max_iterations", 3)):
            reconstruction_raw = self._do_llm_call("Reconstruct code conservatively from an explanation.", up.RCU_RECONSTRUCT.format(LANGUAGE=reconstruction_language, EXPLANATION=explanation), task_id, args.eval_model, f"{cache_pt}/reconstruct_{i+1}", force=force, model_name_override=args.eval_model, cache_model_folder=source_model)
            reconstruction = parse_json(reconstruction_raw, default={})
            reconstructed_code = reconstruction.get("code", "") if isinstance(reconstruction, dict) else ""
            if not reconstructed_code: break
            reconstructed_facts_raw = self._do_llm_call(EVAL_SYS, up.RCU_FACTS.format(CODE=reconstructed_code), task_id, args.eval_model, f"{cache_pt}/reconstructed_facts_{i+1}", force=force, model_name_override=args.eval_model, cache_model_folder=source_model)
            reconstructed_facts = parse_json(reconstructed_facts_raw, default={})
            compare_raw = self._do_llm_call(EVAL_SYS, up.RCU_COMPARE.format(ORIGINAL=json.dumps(original_facts), RECONSTRUCTED=json.dumps(reconstructed_facts)), task_id, args.eval_model, f"{cache_pt}/compare_{i+1}", force=force, model_name_override=args.eval_model, cache_model_folder=source_model)
            comparison = parse_json(compare_raw, default={})
            discrepancies = comparison.get("discrepancies", []) if isinstance(comparison, dict) else []
            required_discrepancy_fields = {
                "original_decision", "original_evidence", "reconstructed_decision",
                "reconstructed_evidence", "valid_input_witness", "observable_difference",
                "smallest_missing_explanation_fact",
            }
            discrepancies = [
                d for d in discrepancies
                if isinstance(d, dict)
                and required_discrepancy_fields.issubset(d)
                and all(str(d[field]).strip() for field in required_discrepancy_fields)
            ]
            iterations.append({"reconstruction": reconstruction, "discrepancies": discrepancies})
            if not discrepancies: break
            patch_raw = self._do_llm_call("Apply only material implementation facts to an explanation.", up.RCU_PATCH.format(EXPLANATION=explanation, DISCREPANCIES=json.dumps(discrepancies, indent=2)), task_id, args.eval_model, f"{cache_pt}/patch_{i+1}", force=force, model_name_override=args.eval_model, cache_model_folder=source_model)
            patch = parse_json(patch_raw, default={})
            updated = normalize_explanation(patch.get("explanation", "") if isinstance(patch, dict) else "")
            if not updated or updated == explanation: break
            explanation = updated
        styled = self._do_llm_call("Faithfully align explanation terminology to the task without changing behavior.", up.HIRE_EXPLAINER_STYLE_TRANSFER.format(PROBLEM=problem_prompt, EXPLANATION=explanation), task_id, args.eval_model, f"{cache_pt}/style_transfer", force=force, model_name_override=args.eval_model, cache_model_folder=source_model)
        final_prompt = up.HIRE_EXPLAINER_ALIGNMENT_CHECKER.format(PROBLEM=problem_prompt, EXPLANATION=styled)
        verdict = self._do_llm_call(HIRE_EXPLAINER_SYS, final_prompt, task_id, args.eval_model, cache_pt, force=force, model_name_override=args.eval_model, cache_model_folder=source_model)
        result = {"correct": (parse_json(verdict, default={}) or {}).get("correct", False), "refined_explanation": explanation, "style_transferred_explanation": styled, "iterations": iterations, "final_judgment": verdict}
        trace=json.dumps(result, indent=2); os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"w",encoding="utf-8") as f: json.dump({"content":trace},f)
        return trace

    def run_rcu_fact_coverage(self, task_id, problem_prompt, cleaned_code, prompt_type, force=False):
        """Refine an explanation until its stated facts cover code-grounded behavior."""
        from utils.mcts_judge import parse_json

        def normalize_explanation(value):
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, dict):
                return "\n\n".join(
                    f"{str(key).replace('_', ' ').capitalize()}:\n{rendered}"
                    for key, item in value.items()
                    if (rendered := normalize_explanation(item))
                )
            if isinstance(value, (list, tuple)):
                return "\n\n".join(
                    f"- {rendered}" for item in value
                    if (rendered := normalize_explanation(item))
                )
            return "" if value is None else str(value).strip()

        args = self.args
        preserving = "preserving" in prompt_type
        source_model = getattr(args, "explainer_model", None) or args.eval_model
        cache_pt = prompt_type + (f"_lambda_L{args.lambda_val}" if args.lambda_val else "")
        path = os.path.join(self.cache_dir, args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(source_model), _sanitize_filename(cache_pt), f"{_sanitize_filename(task_id)}.json")
        if not force and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("content", "")

        source = f"hire_explainer_obj_lambda_L{args.lambda_val}" if args.lambda_val else "hire_explainer_obj"
        explanation = normalize_explanation(self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=False))
        code_facts_raw = self._do_llm_call(
            EVAL_SYS, up.RCU_FACTS.format(CODE=cleaned_code), task_id, args.eval_model,
            f"{cache_pt}/code_facts", force=force, model_name_override=args.eval_model,
            cache_model_folder=source_model,
        )
        code_facts = parse_json(code_facts_raw, default={})
        if not isinstance(code_facts, dict):
            code_facts = {"decisions": []}
        for index, fact in enumerate(code_facts.get("decisions", [])):
            if isinstance(fact, dict):
                fact.setdefault("fact_id", f"F{index + 1}")

        iterations = []
        previous_signature = None
        for i in range(getattr(args, "rcu_max_iterations", 3)):
            explanation_facts_raw = self._do_llm_call(
                EVAL_SYS, up.RCU_EXPLANATION_FACTS.format(EXPLANATION=explanation), task_id,
                args.eval_model, f"{cache_pt}/explanation_facts_{i+1}", force=force,
                model_name_override=args.eval_model, cache_model_folder=source_model,
            )
            explanation_facts = parse_json(explanation_facts_raw, default={})
            coverage_raw = self._do_llm_call(
                EVAL_SYS, up.RCU_FACT_COVERAGE.format(
                    CODE_FACTS=json.dumps(code_facts, indent=2),
                    EXPLANATION_FACTS=json.dumps(explanation_facts, indent=2),
                ), task_id, args.eval_model, f"{cache_pt}/fact_coverage_{i+1}", force=force,
                model_name_override=args.eval_model, cache_model_folder=source_model,
            )
            coverage = parse_json(coverage_raw, default={})
            missing = coverage.get("missing_facts", []) if isinstance(coverage, dict) else []
            required = {"code_fact_id", "code_fact", "code_evidence", "status", "observable_consequence", "minimal_fact_to_add"}
            missing = [fact for fact in missing if isinstance(fact, dict) and required.issubset(fact) and all(str(fact[k]).strip() for k in required)]
            signature = tuple(sorted((str(f["code_fact_id"]), str(f["status"]), str(f["minimal_fact_to_add"])) for f in missing))
            iterations.append({"explanation_facts": explanation_facts, "coverage": coverage, "missing_facts": missing})
            if not missing or signature == previous_signature:
                break
            previous_signature = signature
            patch_raw = self._do_llm_call(
                "Add only verified missing implementation facts to the explanation.",
                up.RCU_FACT_COVERAGE_PATCH.format(EXPLANATION=explanation, MISSING_FACTS=json.dumps(missing, indent=2)),
                task_id, args.eval_model, f"{cache_pt}/patch_{i+1}", force=force,
                model_name_override=args.eval_model, cache_model_folder=source_model,
            )
            patch = parse_json(patch_raw, default={})
            updated = normalize_explanation(patch.get("explanation", "") if isinstance(patch, dict) else "")
            if not updated or updated == explanation:
                break
            explanation = updated

        styled = self._do_llm_call(
            "Faithfully align explanation terminology to the task without changing behavior.",
            up.HIRE_EXPLAINER_STYLE_TRANSFER.format(PROBLEM=problem_prompt, EXPLANATION=explanation),
            task_id, args.eval_model, f"{cache_pt}/style_transfer", force=force,
            model_name_override=args.eval_model, cache_model_folder=source_model,
        )
        verdict = self._do_llm_call(
            HIRE_EXPLAINER_SYS, up.HIRE_EXPLAINER_ALIGNMENT_CHECKER.format(PROBLEM=problem_prompt, EXPLANATION=styled),
            task_id, args.eval_model, cache_pt, force=force, model_name_override=args.eval_model,
            cache_model_folder=source_model,
        )
        final = parse_json(verdict, default={}) or {}
        result = {
            "correct": final.get("correct", False), "refined_explanation": explanation,
            "style_transferred_explanation": styled, "code_facts": code_facts,
            "iterations": iterations, "final_judgment": verdict,
        }
        trace = json.dumps(result, indent=2)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def run_pre_style_refine_post_style_audit(self, task_id, problem_prompt, cleaned_code, prompt_type, force=False):
        """Code-faithfulness checks before and after task-facing style transfer."""
        from utils.mcts_judge import parse_json

        def explanation_from(response, fallback):
            parsed = parse_json(response, default={})
            value = parsed.get("explanation", "") if isinstance(parsed, dict) else ""
            return value.strip() if isinstance(value, str) and value.strip() else fallback

        args = self.args
        preserving = "preserving" in prompt_type
        source_model = getattr(args, "explainer_model", None) or args.eval_model
        cache_pt = prompt_type + (f"_lambda_L{args.lambda_val}" if args.lambda_val else "")
        path = os.path.join(self.cache_dir, args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(source_model), _sanitize_filename(cache_pt), f"{_sanitize_filename(task_id)}.json")
        if not force and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("content", "")

        raw_source = f"hire_explainer_obj_lambda_L{args.lambda_val}" if args.lambda_val else "hire_explainer_obj"
        raw_explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, raw_source, force=False)
        pre_raw = self._do_llm_call(
            EVAL_SYS, (up.CODE_EXPLANATION_SELF_REFINE_PRESERVING if preserving else up.CODE_EXPLANATION_SELF_REFINE).format(CODE=cleaned_code, EXPLANATION=raw_explanation),
            task_id, args.eval_model, f"{cache_pt}/pre_style_refine", force=force,
            model_name_override=args.eval_model, cache_model_folder=source_model,
        )
        refined = explanation_from(pre_raw, raw_explanation)
        styled = self._do_llm_call(
            "Faithfully align explanation terminology to the task without changing behavior.",
            up.HIRE_EXPLAINER_STYLE_TRANSFER.format(PROBLEM=problem_prompt, EXPLANATION=refined),
            task_id, args.eval_model, f"{cache_pt}/style_transfer", force=force,
            model_name_override=args.eval_model, cache_model_folder=source_model,
        )
        post_raw = self._do_llm_call(
            EVAL_SYS, (up.STYLE_TRANSFER_CODE_AUDIT_PRESERVING if preserving else up.STYLE_TRANSFER_CODE_AUDIT).format(CODE=cleaned_code, EXPLANATION=styled),
            task_id, args.eval_model, f"{cache_pt}/post_style_audit", force=force,
            model_name_override=args.eval_model, cache_model_folder=source_model,
        )
        audited = explanation_from(post_raw, styled)
        verdict = self._do_llm_call(
            HIRE_EXPLAINER_SYS, up.HIRE_EXPLAINER_ALIGNMENT_CHECKER.format(PROBLEM=problem_prompt, EXPLANATION=audited),
            task_id, args.eval_model, cache_pt, force=force,
            model_name_override=args.eval_model, cache_model_folder=source_model,
        )
        final = parse_json(verdict, default={}) or {}
        result = {"correct": final.get("correct", False), "raw_explanation": raw_explanation,
                  "pre_style_refined_explanation": refined, "style_transferred_explanation": styled,
                  "post_style_audited_explanation": audited, "pre_style_refine": pre_raw,
                  "post_style_audit": post_raw, "final_judgment": verdict}
        trace = json.dumps(result, indent=2)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def run_dynamic_detail_style_transfer(self, task_id, problem_prompt, cleaned_code, prompt_type, force=False):
        """Add a small code-grounded risk-detail appendix before style transfer."""
        from utils.mcts_judge import parse_json
        args = self.args
        source_model = getattr(args, "explainer_model", None) or args.eval_model
        cache_pt = prompt_type + (f"_lambda_L{args.lambda_val}" if args.lambda_val else "")
        path = os.path.join(self.cache_dir, args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(source_model), _sanitize_filename(cache_pt), f"{_sanitize_filename(task_id)}.json")
        if not force and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("content", "")

        raw_source = f"hire_explainer_obj_lambda_L{args.lambda_val}" if args.lambda_val else "hire_explainer_obj"
        explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, raw_source, force=False)
        details_raw = self._do_llm_call(
            EVAL_SYS, up.DYNAMIC_DETAIL_EXTRACTOR.format(CODE=cleaned_code), task_id,
            args.eval_model, f"{cache_pt}/dynamic_details", force=force,
            model_name_override=args.eval_model, cache_model_folder=source_model,
        )
        details_obj = parse_json(details_raw, default={}) or {}
        details = details_obj.get("details", []) if isinstance(details_obj, dict) else []
        details = [d for d in details if isinstance(d, dict) and str(d.get("fact", "")).strip() and str(d.get("code_evidence", "")).strip()]
        appendix = "\n\nDecision-critical implementation details:\n" + "\n".join(
            f"- {d['fact']} (code: {d['code_evidence']})" for d in details
        ) if details else ""
        augmented = explanation + appendix
        styled = self._do_llm_call(
            "Faithfully align explanation terminology to the task without changing behavior.",
            up.HIRE_EXPLAINER_STYLE_TRANSFER.format(PROBLEM=problem_prompt, EXPLANATION=augmented), task_id,
            args.eval_model, f"{cache_pt}/style_transfer", force=force,
            model_name_override=args.eval_model, cache_model_folder=source_model,
        )
        verdict = self._do_llm_call(
            HIRE_EXPLAINER_SYS, up.HIRE_EXPLAINER_ALIGNMENT_CHECKER.format(PROBLEM=problem_prompt, EXPLANATION=styled),
            task_id, args.eval_model, cache_pt, force=force, model_name_override=args.eval_model,
            cache_model_folder=source_model,
        )
        final = parse_json(verdict, default={}) or {}
        trace = json.dumps({"correct": final.get("correct", False), "raw_explanation": explanation,
                            "dynamic_details": details, "augmented_explanation": augmented,
                            "style_transferred_explanation": styled, "final_judgment": verdict}, indent=2)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def verify_requirement_interrogation(self, task_id, requirement, valid_domain, probe, behavior_answer,
                                         cache_prompt_type, force=False):
        """Requirement-side verifier for a problem-blind code-behavior interrogation."""
        from utils.mcts_judge import parse_json
        response = self._do_llm_call(
            EVAL_SYS,
            up.REQUIREMENT_INTERROGATION_VERIFIER.format(
                REQUIREMENT=requirement, VALID_DOMAIN=valid_domain, PROBE=probe, BEHAVIOR=behavior_answer,
            ),
            task_id, self.args.eval_model, cache_prompt_type, force=force,
            model_name_override=self.args.eval_model,
        )
        verdict = parse_json(response, default={})
        if not isinstance(verdict, dict) or str(verdict.get("status", "")).lower() not in {"satisfied", "violated", "uncertain"}:
            return {"status": "uncertain", "reasoning": "Verifier returned no valid supported classification.", "raw": response}
        verdict["status"] = verdict["status"].lower()
        verdict["raw"] = response
        return verdict

    def run_requirement_mapped_appeal(self, task_id, problem_prompt, cleaned_code,
                                      prompt_type, force=False):
        """Single explanation judge whose negative findings must cite cached requirement IDs."""
        from utils.mcts_judge import parse_json

        args = self.args
        is_specifications = "specifications_mapped_appeal" in prompt_type
        is_evidence_appeal = "specifications_mapped_appeal_v3" in prompt_type
        is_original_judge_appeal = "original_judge_evidence_appeal" in prompt_type
        if is_original_judge_appeal:
            is_specifications = True
            is_evidence_appeal = True
        source_model = getattr(args, "explainer_model", None) or args.eval_model
        style_model = getattr(args, "style_transfer_model", None) or source_model
        model_folder = f"{source_model}_{style_model}" if style_model != source_model else source_model
        cache_pt = prompt_type
        if args.lambda_val and f"_L{args.lambda_val}" not in cache_pt:
            cache_pt = f"{cache_pt}_lambda_L{args.lambda_val}"
        cache_path = os.path.join(self.cache_dir, args.dataset, _sanitize_filename(self.model_or_source),
                                  _sanitize_filename(model_folder), _sanitize_filename(cache_pt),
                                  f"{_sanitize_filename(task_id)}.json")
        if not force and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f).get("content", "")

        requirements_model = (getattr(args, "specifications_model", None) if is_specifications else getattr(args, "requirements_model", None)) or args.eval_model
        artifact_folder = "specifications" if is_specifications else "requirements"
        requirements_path = os.path.join(self.cache_dir, args.dataset, artifact_folder,
                                         _sanitize_filename(requirements_model), f"{_sanitize_filename(task_id)}.json")
        extractor_prompt = (up.HIRE_SPECIFICATION_EXTRACTOR_USER if is_specifications else up.HIRE_REQUIREMENTS_EXTRACTOR_USER).format(PROBLEM=problem_prompt)
        requirements_response = self._do_llm_call(
            "You extract authoritative functional requirements without adding unstated obligations.",
            extractor_prompt, task_id, requirements_model,
            "specification_extractor" if is_specifications else "requirements_extractor",
            force=getattr(args, "force_specifications", False) if is_specifications else getattr(args, "force_requirements", False),
            model_name_override=requirements_model, cache_path_override=requirements_path,
        )
        requirements_data = parse_json(requirements_response, default={})
        if is_specifications and isinstance(requirements_data, dict):
            requirements = requirements_data.get("functional_obligations", []) + requirements_data.get("output_requirements", [])
            requirements_text = json.dumps(requirements_data, indent=2)
        else:
            requirements = requirements_data.get("requirements", []) if isinstance(requirements_data, dict) else []
            requirements_text = json.dumps(requirements, indent=2)
        requirements = [r for r in requirements if isinstance(r, dict) and r.get("id")]
        valid_ids = {str(r["id"]) for r in requirements}

        if is_original_judge_appeal:
            base_prompt_type = prompt_type.replace("_original_judge_evidence_appeal", "")
            original_response = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, base_prompt_type, force=False)
            original_judgment = parse_json(original_response, default={})
            original_value = original_judgment.get("correct", False) if isinstance(original_judgment, dict) else False
            original_correct = original_value.strip().lower() in ("true", "yes", "1") if isinstance(original_value, str) else bool(original_value)
            if original_correct:
                judge = original_judgment
            else:
                mapper_response = self._do_llm_call(
                    "Faithfully map stated failure reasons to an authoritative specification without adding failures.",
                    up.STYLE_TRANSFER_NEGATIVE_JUDGMENT_MAPPER.format(
                        PROBLEM=problem_prompt, SPECIFICATION=requirements_text,
                        JUDGMENT=json.dumps(original_judgment, indent=2),
                    ), task_id, args.eval_model, f"{cache_pt}/negative_judgment_mapper", force=force,
                    model_name_override=args.eval_model, cache_model_folder=model_folder,
                )
                mapped = parse_json(mapper_response, default={})
                mapped_entries = mapped.get("failure_mappings", []) if isinstance(mapped, dict) else []
                judge = {"correct": False, "reasoning": original_judgment.get("reasoning", "") if isinstance(original_judgment, dict) else original_response,
                         "violations": mapped_entries, "original_judgment": original_judgment}
        else:
            source = f"hire_explainer{'_obj' if 'explainer_obj' in prompt_type else ''}_lambda_L{args.lambda_val}_style_transfer"
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=False)
            judge_response = self._do_llm_call(
                HIRE_EXPLAINER_SYS,
                (up.STYLE_TRANSFER_SPECIFICATION_MAPPED_JUDGE.format(SPECIFICATION=requirements_text, EXPLANATION=explanation)
                 if is_specifications else up.STYLE_TRANSFER_REQUIREMENT_MAPPED_JUDGE.format(REQUIREMENTS=requirements_text, EXPLANATION=explanation)),
                task_id, args.eval_model, f"{cache_pt}/judge_decision", force=force,
                model_name_override=args.eval_model, cache_model_folder=model_folder,
            )
            judge = parse_json(judge_response, default={})
        first_value = judge.get("correct", False) if isinstance(judge, dict) else False
        first_correct = first_value.strip().lower() in ("true", "yes", "1") if isinstance(first_value, str) else bool(first_value)
        raw_violations = judge.get("violations", []) if isinstance(judge, dict) else []
        violations = [v for v in raw_violations if isinstance(v, dict) and str(v.get("requirement_id", "")) in valid_ids]
        output = {"correct": first_correct, "verdict": "Yes" if first_correct else "No",
                  "requirements": requirements, "first_judgment": judge,
                  "mapped_violations": violations, "requirements_appeal": None}
        if not first_correct:
            appeal_response = self._do_llm_call(
                "Audit whether each cited behavior violates its cited authoritative requirement.",
                (up.STYLE_TRANSFER_SPECIFICATION_EVIDENCE_APPEAL.format(PROBLEM=problem_prompt, SPECIFICATION=requirements_text, JUDGE_DECISION=json.dumps(judge, indent=2))
                 if is_evidence_appeal else up.STYLE_TRANSFER_SPECIFICATION_MAPPING_APPEAL.format(SPECIFICATION=requirements_text, JUDGE_DECISION=json.dumps(judge, indent=2))
                 if is_specifications else up.STYLE_TRANSFER_REQUIREMENT_MAPPING_APPEAL.format(REQUIREMENTS=requirements_text, JUDGE_DECISION=json.dumps(judge, indent=2))),
                task_id, args.eval_model, f"{cache_pt}/requirements_mapping_appeal", force=force,
                model_name_override=args.eval_model, cache_model_folder=model_folder,
            )
            appeal = parse_json(appeal_response, default={})
            output["requirements_appeal"] = appeal
            appeal_violations = [
                v for v in appeal.get("violations", [])
                if isinstance(v, dict) and str(v.get("requirement_id", "")) in valid_ids
            ] if isinstance(appeal, dict) else []
            if is_evidence_appeal:
                complete_evidence = all(
                    isinstance(v.get("text_evidence"), list)
                    and isinstance(v.get("example_evidence"), list)
                    and bool(str(v.get("strongest_joint_interpretation", "")).strip())
                    and str(v.get("classification", "")).lower() in {
                        "explicitly_supported", "logically_implied", "ambiguous", "unsupported"
                    }
                    and isinstance(v.get("does_observed_behavior_violate_it"), bool)
                    for v in appeal_violations
                )
                supported = [
                    v for v in appeal_violations
                    if str(v.get("classification", "")).lower() in {"explicitly_supported", "logically_implied"}
                    and v.get("does_observed_behavior_violate_it") is True
                ]
                can_overturn = (violations and len(appeal_violations) == len(violations)
                                and complete_evidence and not supported)
            else:
                statuses = [str(v.get("status", "uncertain")).lower() for v in appeal_violations]
                can_overturn = (violations and len(appeal_violations) == len(violations)
                                and all(status == "unsupported" for status in statuses))
            # Empty, malformed, partial, or uncertain appeals uphold the first
            # verdict.  An overturn needs a complete audit of every mapped claim.
            if can_overturn:
                output["correct"], output["verdict"] = True, "Yes"
        trace = json.dumps(output, indent=2)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def run_single_judge_requirements_appeal(self, task_id, problem_prompt, cleaned_code,
                                              prompt_type, force=False):
        """Appeal a negative explanation-only alignment judgment without using code.

        This is deliberately independent of the dialogue and dual-agent pipelines.
        The base style-transfer judgment remains the sole evaluator; the appeal only
        tests whether its stated failure requirements belong to the problem.
        """
        from utils.mcts_judge import parse_json

        args = self.args
        is_problem_anchored = any(version in prompt_type for version in (
            "requirements_appeal_v2", "requirements_appeal_v3"
        ))
        suffix = "_requirements_appeal_v3" if "requirements_appeal_v3" in prompt_type else "_requirements_appeal_v2" if "requirements_appeal_v2" in prompt_type else "_requirements_appeal"
        base_prompt_type = prompt_type.replace(suffix, "")
        source_model = getattr(args, "explainer_model", None) or args.eval_model
        style_model = getattr(args, "style_transfer_model", None) or source_model
        model_folder = f"{source_model}_{style_model}" if style_model != source_model else source_model

        cache_pt = prompt_type
        if args.lambda_val:
            lambda_suffix = f"_L{args.lambda_val}"
            if lambda_suffix not in cache_pt:
                cache_pt = f"{cache_pt}_lambda{lambda_suffix}" if "_lambda" not in cache_pt else f"{cache_pt}{lambda_suffix}"
        cache_path = os.path.join(
            self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source),
            _sanitize_filename(model_folder), _sanitize_filename(cache_pt),
            f"{_sanitize_filename(task_id)}.json"
        )
        if not force and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f).get("content", "")

        # This is the existing explanation-only style-transfer L3 judgment.  It
        # sees no code and has no dialogue or independent evaluator attached.
        first_response = self.evaluate_prompt(
            task_id, problem_prompt, cleaned_code, base_prompt_type, force=False
        )
        first_judgment = parse_json(first_response, default=None)
        first_correct = False
        if isinstance(first_judgment, dict):
            value = first_judgment.get("correct", False)
            first_correct = value.strip().lower() in ("true", "yes", "1") if isinstance(value, str) else bool(value)

        output = {
            "correct": first_correct,
            "verdict": "Yes" if first_correct else "No",
            "first_judgment": first_judgment if isinstance(first_judgment, dict) else {"raw": first_response},
            "alleged_violated_requirements": None,
            "requirements_appeal": None,
        }

        if not first_correct:
            extractor_model = (
                getattr(args, "requirements_appeal_extractor_model", None)
                or getattr(args, "appeal_requirements_extractor_model", None)
                or args.eval_model
            )
            extractor_template = (
                up.STYLE_TRANSFER_V2_VIOLATED_REQUIREMENTS_EXTRACTOR
                if is_problem_anchored else up.DUAL_AGENT_VIOLATED_REQUIREMENTS_EXTRACTOR
            )
            extractor_prompt = extractor_template.replace(
                "{{evaluator_judgment}}", json.dumps(output["first_judgment"], indent=2)
            )
            extractor_prompt = extractor_prompt.replace("{{problem_statement}}", problem_prompt)
            extractor_response = self._do_llm_call(
                "Extract only code-free alleged behavioral requirements from a judgment.",
                extractor_prompt, task_id, extractor_model,
                f"{cache_pt}/violated_requirements_extractor_{_sanitize_filename(extractor_model)}",
                force=force, model_name_override=extractor_model, cache_model_folder=model_folder,
            )
            extracted = parse_json(extractor_response, default=None)
            raw_requirements = extracted.get("alleged_violated_requirements", []) if isinstance(extracted, dict) else []
            code_pattern = re.compile(r"[`{}\[\]()]|\b(?:def|class)\b|==|!=|<=|>=|=>|->")
            clean_requirements = []
            if isinstance(raw_requirements, list):
                for item in raw_requirements:
                    if not isinstance(item, dict):
                        continue
                    requirement = str(item.get("requirement", "")).strip()
                    basis = str(item.get("judgment_basis", "")).strip()
                    anchor = str(item.get("problem_anchor", "")).strip()
                    has_required_anchor = (not is_problem_anchored) or bool(anchor)
                    if requirement and "\n" not in requirement and not code_pattern.search(requirement) and has_required_anchor:
                        normalized = {"requirement": requirement, "judgment_basis": basis}
                        if is_problem_anchored:
                            normalized["problem_anchor"] = anchor
                        clean_requirements.append(normalized)
            alleged_requirements = {"alleged_violated_requirements": clean_requirements}
            output["alleged_violated_requirements"] = alleged_requirements

            if clean_requirements:
                appeal_prompt = (
                    up.STYLE_TRANSFER_V2_EXTRACTED_REQUIREMENTS_APPEAL
                    if is_problem_anchored else up.DUAL_AGENT_EXTRACTED_REQUIREMENTS_APPEAL
                )
                appeal_prompt = appeal_prompt.replace("{{problem_statement}}", problem_prompt)
                appeal_prompt = appeal_prompt.replace("{{alleged_requirements}}", json.dumps(alleged_requirements, indent=2))
                appeal_response = self._do_llm_call(
                    "Check whether alleged behavioral requirements are actually required by the problem.",
                    appeal_prompt, task_id, args.eval_model,
                    f"{cache_pt}/requirements_appeal_from_extracted_requirements",
                    force=force, model_name_override=args.eval_model, cache_model_folder=model_folder,
                )
                appeal = parse_json(appeal_response, default=None)
                output["requirements_appeal"] = appeal
                statuses = [
                    str(item.get("status", "uncertain")).strip().lower()
                    for item in appeal.get("requirements", [])
                    if isinstance(item, dict)
                ] if isinstance(appeal, dict) else []
                # An incomplete or uncertain audit never changes a negative verdict.
                if statuses and len(statuses) == len(clean_requirements) and all(status == "not_required" for status in statuses):
                    output["correct"] = True
                    output["verdict"] = "Yes"

        trace = json.dumps(output, indent=2)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def evaluate_prompt(self, task_id, problem_prompt, cleaned_code, prompt_type, force=False):
        args = self.args
        if "dynamic_detail_style_transfer" in prompt_type:
            return self.run_dynamic_detail_style_transfer(task_id, problem_prompt, cleaned_code, prompt_type, force=force)
        if "pre_style_refine_post_style_audit" in prompt_type:
            return self.run_pre_style_refine_post_style_audit(task_id, problem_prompt, cleaned_code, prompt_type, force=force)
        if "rcu_fact_coverage" in prompt_type:
            return self.run_rcu_fact_coverage(task_id, problem_prompt, cleaned_code, prompt_type, force=force)
        if "rcu_refine" in prompt_type:
            return self.run_rcu_refine(task_id, problem_prompt, cleaned_code, prompt_type, force=force)
        if prompt_type.startswith("vanilla_evidence_appeal") or prompt_type.startswith("codejudge_evidence_appeal"):
            return self.run_direct_judge_evidence_appeal(task_id, problem_prompt, cleaned_code, prompt_type, force=force)
        if ("requirements_mapped_appeal" in prompt_type or "specifications_mapped_appeal" in prompt_type
                or "original_judge_evidence_appeal" in prompt_type):
            return self.run_requirement_mapped_appeal(task_id, problem_prompt, cleaned_code, prompt_type, force=force)
        if "alignment_checker" in prompt_type and "requirements_appeal" in prompt_type:
            return self.run_single_judge_requirements_appeal(
                task_id, problem_prompt, cleaned_code, prompt_type, force=force
            )
        source_model = getattr(args, 'explainer_model', None) or args.eval_model
        active_system_prompt = EVAL_SYS
        if "explainer" in prompt_type and "checker" not in prompt_type: active_system_prompt = EXPLAINER_SYS
        elif "decomposer" in prompt_type or "commentor" in prompt_type: active_system_prompt = EXPLAINER_SYS
        elif "hire_pseudo_checker" in prompt_type: active_system_prompt = HIRE_PSEUDO_SYS
        elif "hire_explainer_checker" in prompt_type or "hire_explainer_alignment_checker" in prompt_type:
            active_system_prompt = HIRE_EXPLAINER_SYS_FAITHFUL if "faithful" in prompt_type else HIRE_EXPLAINER_SYS
        
        is_query_aware = "query_aware" in prompt_type
        eval_user_prompt = ""
        first_judgment_for_reconsideration = None
        
        # Dependency Resolution
        if prompt_type == "cj_summary":
            content = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, "cj_analysis", force=False)
            eval_user_prompt = up.CODEJUDGE_SUMMARY.format(ANALYSIS=content)
        elif prompt_type == "cj_analysis":
            eval_user_prompt = up.CODEJUDGE_ANALYSIS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif prompt_type == "vanilla":
            eval_user_prompt = up.VANILLA_EVAL_BINARY.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif prompt_type == "vanilla_no_reasoning":
            eval_user_prompt = up.VANILLA_EVAL_BINARY_NO_REASONING.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif prompt_type == "behavior_comparison":
            eval_user_prompt = up.BEHAVIOR_COMPARISON.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif prompt_type == "behavior_comparison_no_rc":
            eval_user_prompt = up.BEHAVIOR_COMPARISON_NO_RC.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif "behavior_comparison_explanation" in prompt_type:
            source = f"hire_explainer{'_obj' if 'obj' in prompt_type else ''}{'_query_aware' if is_query_aware else ''}"
            if args.lambda_val: source = f"{source}_lambda_L{args.lambda_val}"
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=False)
            eval_user_prompt = up.BEHAVIOR_COMPARISON_EXPLANATION.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
        elif prompt_type == "two_phase_reflective":
            eval_user_prompt = up.TWO_PHASE_REFLECTIVE_EVAL.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif "two_phase_reflective_explanation" in prompt_type:
            source = f"hire_explainer{'_obj' if 'obj' in prompt_type else ''}{'_query_aware' if is_query_aware else ''}"
            if args.lambda_val: source = f"{source}_lambda_L{args.lambda_val}"
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=False)
            eval_user_prompt = up.TWO_PHASE_REFLECTIVE_EVAL_EXPLANATION.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
        elif "style_transfer" in prompt_type and "checker" not in prompt_type:
            is_specification_aware_style = "specification_aware" in prompt_type
            style_suffix = "_style_transfer_specification_aware" if is_specification_aware_style else "_style_transfer"
            base_prompt = prompt_type.replace(style_suffix, "")
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, base_prompt, force=False)
            if is_specification_aware_style:
                specification_model = getattr(args, "specifications_model", None) or args.eval_model
                specification_path = os.path.join(
                    self.cache_dir, args.dataset, "specifications", _sanitize_filename(specification_model),
                    f"{_sanitize_filename(task_id)}.json"
                )
                specification = self._do_llm_call(
                    "You extract authoritative structured specifications without adding unstated obligations.",
                    up.HIRE_SPECIFICATION_EXTRACTOR_USER.format(PROBLEM=problem_prompt), task_id,
                    specification_model, "specification_extractor",
                    force=getattr(args, "force_specifications", False),
                    model_name_override=specification_model, cache_path_override=specification_path,
                )
                eval_user_prompt = up.HIRE_EXPLAINER_STYLE_TRANSFER_SPECIFICATION_AWARE.format(
                    PROBLEM=problem_prompt, SPECIFICATION=specification, EXPLANATION=explanation
                )
            else:
                eval_user_prompt = up.HIRE_EXPLAINER_STYLE_TRANSFER.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
            active_system_prompt = "You are an expert technical editor. Your goal is to rewrite a code explanation to semantically align it with a problem statement. Crucially, the explanation is for a potentially buggy/incorrect code implementation: you MUST strictly preserve all logical bugs, incorrect execution details, and incorrect walkthrough outputs described in the original explanation."
        elif "hire_explainer" in prompt_type and "checker" not in prompt_type:
            base = prompt_type.replace("_lambda", "")
            suffix = "_FAITHFUL" if "_faithful" in base else "_NO_WT" if "_no_wt" in base else ""
            if suffix: base = base.replace(suffix.lower(), "")
            if args.lambda_val:
                obj_p = "OBJ_" if "explainer_obj" in base else ""
                prompt_key = f"HIRE_EXPLAINER_{obj_p}L{args.lambda_val}{'_QUERY_AWARE' if 'query_aware' in base else ''}{suffix}"
            else: prompt_key = f"{base.upper()}{suffix}"
            prompt_template = getattr(up, prompt_key, None) or getattr(up, prompt_key.replace(suffix, ""), None)
            if not prompt_template:
                is_obj = "explainer_obj" in prompt_type
                prompt_template = up.HIRE_EXPLAINER_OBJ_QUERY_AWARE if (is_obj and "query_aware" in prompt_type) else up.HIRE_EXPLAINER_QUERY_AWARE if "query_aware" in prompt_type else up.HIRE_EXPLAINER_OBJ if is_obj else up.HIRE_EXPLAINER
            if suffix == "_FAITHFUL": prompt_template = prompt_template.replace("Describe exactly what the code does", "Describe exactly what the code does. The explanation must be FAITHFUL to the implementation; avoid any 'consistency hallucinations' where you describe what the code SHOULD do instead of what it ACTUALLY does.")
            elif suffix == "_NO_WT": prompt_template = '\n'.join([l for l in prompt_template.split('\n') if "walkthrough" not in l.lower()])
            eval_user_prompt = prompt_template.format(PROBLEM=problem_prompt, CODE=cleaned_code) if "query_aware" in prompt_type else prompt_template.format(CODE=cleaned_code)
        elif "hire_pseudo" in prompt_type and "checker" not in prompt_type:
            base = prompt_type.replace("_lambda", "")
            prompt_key = f"HIRE_PSEUDO_L{args.lambda_val}{'_QUERY_AWARE' if 'query_aware' in base else ''}" if args.lambda_val else base.upper()
            prompt_template = getattr(up, prompt_key, None) or (up.HIRE_PSEUDO_QUERY_AWARE if "query_aware" in prompt_type else up.HIRE_PSEUDO)
            eval_user_prompt = prompt_template.format(PROBLEM=problem_prompt, CODE=cleaned_code) if ("query_aware" in prompt_type or "{PROBLEM}" in prompt_template) else prompt_template.format(CODE=cleaned_code)
        elif "hire_pseudo" in prompt_type and "checker" in prompt_type:
            source = f"hire_pseudo{'_query_aware' if is_query_aware else ''}"
            if args.lambda_val: source = f"{source}_lambda_L{args.lambda_val}"
            ps = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=False)
            eval_user_prompt = up.HIRE_PSEUDO_CHECKER.format(PROBLEM=problem_prompt, PSEUDOCODE=ps)
        elif "hire_explainer" in prompt_type and "checker" in prompt_type and "alignment" not in prompt_type:
            is_query_aware = "query_aware" in prompt_type
            is_objective = "explainer_obj" in prompt_type
            update = "_style_transfer" if "style_transfer" in prompt_type else ""
            source = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
            if args.lambda_val: source = f"{source}_lambda_L{args.lambda_val}"
            source += update
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=force)
            eval_user_prompt = up.HIRE_EXPLAINER_CHECKER.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
        elif "hire_explainer" in prompt_type and "alignment_checker" in prompt_type and "code_reconsideration" in prompt_type:
            first_prompt_type = prompt_type.replace("_code_reconsideration", "")
            first_judgment_for_reconsideration = self.evaluate_prompt(
                task_id, problem_prompt, cleaned_code, first_prompt_type, force=False
            )
            source = f"hire_explainer{'_obj' if 'explainer_obj' in prompt_type else ''}{'_query_aware' if is_query_aware else ''}"
            if args.lambda_val:
                source = f"{source}_lambda_L{args.lambda_val}"
            if "style_transfer" in prompt_type:
                source += "_style_transfer"
            explanation = self.evaluate_prompt(
                task_id, problem_prompt, cleaned_code, source, force=False
            )
            eval_user_prompt = up.STYLE_TRANSFER_CODE_RECONSIDERATION.format(
                PROBLEM=problem_prompt,
                EXPLANATION=explanation,
                FIRST_JUDGMENT=first_judgment_for_reconsideration,
                CODE=cleaned_code,
            )
        elif "hire_explainer" in prompt_type and "alignment_checker" in prompt_type:
            suffix = "_faithful" if "faithful" in prompt_type else "_no_wt" if "no_wt" in prompt_type else ""
            update = "_direct_update" if "direct_update" in prompt_type else "_self_refine" if "self_refine" in prompt_type else "_update" if "update" in prompt_type else ""
            if "style_transfer" in prompt_type:
                update = "_style_transfer_specification_aware" if "specification_aware" in prompt_type else "_style_transfer"
            source = f"hire_explainer{'_obj' if 'explainer_obj' in prompt_type else ''}{'_query_aware' if is_query_aware else ''}{suffix}"
            if args.lambda_val: source = f"{source}_lambda_L{args.lambda_val}"
            source += update
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source, force=force)
            checker_prompt = (up.HIRE_EXPLAINER_ALIGNMENT_CHECKER_REASONING_FIRST if "reasoning_first" in prompt_type
                              else up.HIRE_EXPLAINER_ALIGNMENT_CHECKER_FAITHFUL if "faithful" in prompt_type
                              else up.HIRE_EXPLAINER_ALIGNMENT_CHECKER)
            eval_user_prompt = checker_prompt.format(PROBLEM=problem_prompt, EXPLANATION=explanation)
        elif prompt_type.startswith("hire_decomposer"):
            if "hire_decomposer_flexible" in prompt_type:
                eval_user_prompt = up.HIRE_DECOMPOSER_WITH_PROBLEM_AT_MOST_N.format(PROBLEM=problem_prompt, N=args.n, CODE=cleaned_code) if is_query_aware else up.HIRE_DECOMPOSER_AT_MOST_N.format(N=args.n, CODE=cleaned_code)
            else:
                eval_user_prompt = up.HIRE_DECOMPOSER_WITH_PROBLEM.format(PROBLEM=problem_prompt, N=args.n, CODE=cleaned_code) if is_query_aware else up.HIRE_DECOMPOSER.format(N=args.n, CODE=cleaned_code)
        elif any(x in prompt_type for x in ["hire_plan_checker", "hire_commentor_checker", "hire_implementation_checker", "hire_aggregator"]):
            is_flex = "flexible" in prompt_type
            dec_folder = f"hire_decomposer{'_query_aware' if is_query_aware else ''}{'_flexible' if is_flex else ''}_N_{args.n}"
            plan_content = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, dec_folder, force=False)
            if "hire_plan_checker" in prompt_type:
                plan = plan_content
                if "text_only" in prompt_type:
                    try:
                        js = json.loads(plan[plan.find('{'):plan.rfind('}')+1])
                        plan = json.dumps({"steps": [{"step": i + 1, "explanation": s.get("explanation", "")} for i, s in enumerate(js.get("steps", []))]}, indent=2)
                    except: pass
                eval_user_prompt = up.HIRE_PLAN_CHECKER.format(PROBLEM=problem_prompt, PLAN=plan)
            elif "hire_commentor_checker" in prompt_type:
                try:
                    js = json.loads(plan_content[plan_content.find('{'):plan_content.rfind('}')+1])
                    eval_user_prompt = up.HIRE_COMMENTOR_CODE_CHECKER.format(PROBLEM=problem_prompt, AUGMENTED_CODE=augment_code_with_comments(cleaned_code, js.get("steps", [])))
                except: raise
            elif "hire_aggregator" in prompt_type:
                pc_raw = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, f"hire_plan_checker{'_query_aware_text_only_flexible' if is_query_aware else '_text_only_flexible'}_N_{args.n}", force=False)
                cc_raw = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, f"hire_commentor_checker{'_query_aware_flexible' if is_query_aware else '_flexible'}_N_{args.n}", force=False)
                try:
                    eval_user_prompt = (up.HIRE_AGGREGATOR_A2_AWARE if "a2_aware" in prompt_type else up.HIRE_AGGREGATOR).format(PROBLEM=problem_prompt, CODE=cleaned_code, PLAN_REASONING=json.loads(pc_raw.strip()).get("reasoning", ""), COMMENTOR_REASONING=json.loads(cc_raw.strip()).get("reasoning", ""))
                except: raise
            elif "hire_implementation_checker" in prompt_type:
                logic = "isolated" if "isolated" in prompt_type else "context"
                try: js = json.loads(plan_content[plan_content.find('{'):plan_content.rfind('}')+1])
                except: raise
                prev_ctx, results = "", []
                for idx, step in enumerate(js.get("steps", [])):
                    desc, code = step.get("explanation", ""), step.get("code_segment", "")
                    prompt = up.HIRE_IMPLEMENTATION_CHECKER_ISOLATED.format(STEP_DESC=desc, STEP_CODE=code) if logic=="isolated" else up.HIRE_IMPLEMENTATION_CHECKER_CONTEXT.format(PROBLEM=problem_prompt, PREVIOUS_STEPS=prev_ctx or "None", CURRENT_STEP_DESC=desc, CURRENT_STEP_CODE=code)
                    if logic != "isolated": prev_ctx += f"Step {idx+1}: {desc}\nImplementation:\n{code}\n\n"
                    results.append(self._do_llm_call(active_system_prompt, prompt, task_id, self.model_name, f"{prompt_type}_step_{idx+1}", force))
                return "\n".join(results)
        elif prompt_type == "cj_fault_localization": eval_user_prompt = up.CODEJUDGE_FAULT_LOCALIZATION.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif prompt_type == "ice_correctness": eval_user_prompt = up.ICE_CORRECTNESS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
        elif prompt_type == "ice_usefulness": eval_user_prompt = up.ICE_USEFULNESS.format(PROBLEM=problem_prompt, CODE=cleaned_code)

        if not eval_user_prompt: raise ValueError(f"Unsupported prompt type: {prompt_type}")
        
        # Determine cache prompt type with suffixes
        cache_pt = prompt_type
        is_lambda_aware = any(x in prompt_type for x in ["alignment", "explainer", "pseudo", "explanation"])
        if args.lambda_val and is_lambda_aware:
            l_suffix = f"_L{args.lambda_val}"
            if l_suffix not in prompt_type:
                cache_pt = f"{prompt_type}_lambda{l_suffix}" if "_lambda" not in prompt_type else f"{prompt_type}{l_suffix}"
        elif (prompt_type.startswith("hire_") and all(x not in prompt_type for x in ["explainer", "pseudo", "N_"])):
            n_suffix = f"_N_{args.n}"
            if n_suffix not in prompt_type:
                cache_pt = f"{prompt_type}{n_suffix}"

        # Determine which model folder to use for caching
        is_generator = any(prompt_type.startswith(x) for x in ["hire_explainer", "hire_pseudo", "hire_decomposer"]) and "checker" not in prompt_type
        if prompt_type == "cj_analysis": is_generator = True
        
        target_model_folder = source_model if is_generator else self.model_name
        model_override = None
        if "style_transfer" in prompt_type:
            style_transfer_model = getattr(args, 'style_transfer_model', None) or source_model
            target_model_folder = f"{source_model}_{style_transfer_model}" if style_transfer_model != source_model else source_model
            model_override = style_transfer_model

        content = self._do_llm_call(
            active_system_prompt, eval_user_prompt, task_id, target_model_folder,
            cache_pt, force, model_name_override=model_override
        )
        if "code_reconsideration" in prompt_type:
            from utils.mcts_judge import parse_json
            first_data = parse_json(first_judgment_for_reconsideration, default=None)
            final_data = parse_json(content, default=None)
            if isinstance(first_data, dict) and isinstance(final_data, dict):
                first_correct = first_data.get("correct", False)
                final_correct = final_data.get("correct", False)
                if isinstance(first_correct, str):
                    first_correct = first_correct.strip().lower() in ("true", "yes", "1")
                if isinstance(final_correct, str):
                    final_correct = final_correct.strip().lower() in ("true", "yes", "1")
                changed = bool(first_correct) != bool(final_correct)
                citations = final_data.get("code_citations", [])
                valid_citations = []
                if isinstance(citations, list):
                    for citation in citations:
                        if isinstance(citation, dict):
                            snippet = citation.get("code", "")
                            if isinstance(snippet, str) and snippet and snippet in cleaned_code:
                                valid_citations.append(citation)
                final_data["decision_changed"] = changed
                final_data["code_citations_valid"] = (not changed) or bool(valid_citations)
                if changed and not valid_citations:
                    final_data["correct"] = bool(first_correct)
                    final_data["decision_changed"] = False
                    final_data["reasoning"] = (
                        "The proposed verdict change was rejected because it did not cite an exact "
                        "verbatim code substring. " + str(final_data.get("reasoning", ""))
                    )
                final_data["code_citations"] = valid_citations
                content = json.dumps(final_data, indent=2)
                # Replace the initially cached raw response with the validated trace.
                validated_path = os.path.join(
                    self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source),
                    _sanitize_filename(target_model_folder), _sanitize_filename(cache_pt),
                    f"{_sanitize_filename(task_id)}.json"
                )
                if os.path.exists(validated_path):
                    with open(validated_path, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    cached["content"] = content
                    with open(validated_path, "w", encoding="utf-8") as f:
                        json.dump(cached, f)
        return content

    def run_mode(self, task_id, problem_prompt, row, force=False):
        mode = self.args.mode
        if mode == "code":
            lang = self._resolve_programming_language(row)
            sys_prompt = up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=lang.upper(), PROGRAM_LANGUAGE_LOWER=lang)
            # For code mode, problem_prompt is the user prompt
            model_name = self.args.code_gen_model or "gpt-4o-mini"
            # Note: in code mode, the caching expects dataset/code_gen_model/task_id.json
            # _do_llm_call uses dataset/model_or_source/eval_model/eval_prompt_type/task_id.json
            # To match generate_code_batch exactly, we might need a custom call or adjust _do_llm_call.
            # However, code mode in generate_batch.py is simpler.
            
            # Let's customize cache path for code mode to match generate_code_batch
            cache_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(model_name), f"{_sanitize_filename(task_id)}.json")
            if not force and os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("content", "")
            
            api_model_name = resolve_model(model_name)
            extra_kwargs = {}
            if is_together_model(api_model_name):
                from together import Together
                client = Together()
                response = client.chat.completions.create(
                    model=api_model_name,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": problem_prompt}],
                    **extra_kwargs
                )
                content = response.choices[0].message.content
            elif api_model_name.startswith("gemini-"):
                from openai import OpenAI
                gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not gemini_api_key:
                    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for Gemini models.")
                client = OpenAI(
                    api_key=gemini_api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                if "gemini-3" in api_model_name or "gemini-3.5" in api_model_name or "thinking" in api_model_name:
                    extra_kwargs["reasoning_effort"] = "high"
                response = client.chat.completions.create(
                    model=api_model_name,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": problem_prompt}],
                    **extra_kwargs
                )
                content = response.choices[0].message.content
            elif api_model_name.startswith("claude-"):
                from anthropic import Anthropic
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
                if not anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable is required for Claude models.")
                client = Anthropic(api_key=anthropic_api_key)
                if "opus-4.6" in api_model_name or "sonnet-4.6" in api_model_name or "3-7-sonnet" in api_model_name or "thinking" in api_model_name:
                    extra_kwargs["thinking"] = {"type": "adaptive"}
                    extra_kwargs["max_tokens"] = 16384
                else:
                    extra_kwargs["max_tokens"] = 4096
                response = client.messages.create(
                    model=api_model_name,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": problem_prompt}],
                    **extra_kwargs
                )
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
            else:
                from openai import OpenAI
                client = OpenAI()
                response = client.chat.completions.create(
                    model=api_model_name,
                    messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": problem_prompt}],
                    **extra_kwargs
                )
                content = response.choices[0].message.content
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"content": content}, f)
            return content

        elif mode == "eval":
            raw_code = ""
            if self.args.eval_source:
                raw_code = row.get(self.args.eval_source, "")
            
            if not raw_code:
                # Try loading from cache if eval_source is not in row or is empty
                model_name = self.args.eval_source or self.args.code_gen_model or "gpt-4o-mini"
                cache_path = os.path.join(self.cache_dir, self.args.dataset, model_name, f"{_sanitize_filename(task_id)}.json")
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        raw_code = json.load(f).get("content", "")
            
            if not raw_code: return "Error: No code to evaluate"
            cleaned_code = clean_code(raw_code)
            return self.evaluate_prompt(task_id, problem_prompt, cleaned_code, self.args.eval_prompt, force=force)
        
        elif mode == "refine":
            refine_model = self.args.refine_model or self.args.code_gen_model or "gpt-4o-mini"
            initial_code = row.get(self.args.eval_source, "")
            if not initial_code: return "Error: No initial code"
            
            # Load evaluation from cache (simplified version of generate_refine_batch)
            eval_prompt_type = self.args.eval_prompt
            if self.args.lambda_val is not None:
                suffix = f"_L{self.args.lambda_val}"
                if "_lambda" not in eval_prompt_type: eval_prompt_type = f"{self.args.eval_prompt}_lambda{suffix}"
                elif suffix not in eval_prompt_type: eval_prompt_type = f"{self.args.eval_prompt}{suffix}"
            
            model_folder = f"{self.args.explainer_model}_{self.args.eval_model}" if self.args.explainer_model else self.args.eval_model
            eval_cache_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.args.eval_source), _sanitize_filename(model_folder), _sanitize_filename(eval_prompt_type), f"{_sanitize_filename(task_id)}.json")
            
            if not os.path.exists(eval_cache_path): return "Error: Missing evaluation"
            
            with open(eval_cache_path, 'r', encoding='utf-8') as f:
                evaluation_text = json.load(f).get("content", "")
            
            # Standardize extraction of reasoning
            if '"reasoning":' in evaluation_text:
                try:
                    r_start = evaluation_text.find('"reasoning":') + len('"reasoning":')
                    r_end = evaluation_text.rfind('}')
                    evaluation_text = evaluation_text[r_start:r_end].strip().strip('"').replace('\\n', '\n').replace('\\"', '"')
                except: pass

            refine_user_prompt = up.REFINE_PROMPT.format(PROBLEM=problem_prompt, CODE=initial_code, EVALUATION=evaluation_text)
            lang = self._resolve_programming_language(row)
            sys_prompt = up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=lang.upper(), PROGRAM_LANGUAGE_LOWER=lang)
            
            cache_pt = f"{eval_prompt_type}_refine_{refine_model}"
            return self._do_llm_call(sys_prompt, refine_user_prompt, task_id, refine_model, cache_pt, force)

        elif mode == "reconstruct":
            reconstruct_model = self.args.reconstruct_model or self.args.eval_model or "gpt-4o-mini"
            is_query_aware = "query_aware" in (self.args.eval_prompt or "")
            is_objective = "explainer_obj" in (self.args.eval_prompt or "")
            
            source_prompt = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
            source_folder = f"{source_prompt}_lambda_L{self.args.lambda_val}" if self.args.lambda_val is not None else source_prompt
            
            source_model = self.args.explainer_model or self.args.eval_model
            explainer_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
            
            if not os.path.exists(explainer_path): return "Error: Missing explanation"
            with open(explainer_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f).get("content", "")

            reconstruct_user_prompt = up.HIRE_RECONSTRUCT_QUERY_AWARE.format(PROBLEM=problem_prompt, EXPLANATION=explanation) if is_query_aware else up.HIRE_RECONSTRUCT.format(EXPLANATION=explanation)
            lang = self._resolve_programming_language(row)
            sys_prompt = up.CODEGEN_SYS.format(PROGRAM_LANGUAGE=lang.upper(), PROGRAM_LANGUAGE_LOWER=lang)
            
            l_part = f"_L{self.args.lambda_val}" if self.args.lambda_val is not None else ""
            cache_pt = f"reconstruct_{self.args.eval_prompt}{l_part}"
            return self._do_llm_call(sys_prompt, reconstruct_user_prompt, task_id, f"reconstruct_{reconstruct_model}", cache_pt, force)

        elif mode == "compare":
            compare_model = self.args.compare_model or self.args.eval_model
            original_code = clean_code(row.get(self.args.eval_source or "code", ""))
            
            is_query_aware = "query_aware" in (self.args.eval_prompt or "")
            is_objective = "explainer_obj" in (self.args.eval_prompt or "")
            source_prompt = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
            source_folder = f"{source_prompt}_lambda_L{self.args.lambda_val}" if self.args.lambda_val is not None else source_prompt
            
            source_model = self.args.explainer_model or self.args.eval_model
            explainer_path = os.path.join(self.cache_dir, self.args.dataset, self.model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
            if not os.path.exists(explainer_path): return "Error: Missing explanation"
            with open(explainer_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f).get("content", "")

            reconstruct_model = self.args.reconstruct_model or self.args.eval_model or "gpt-4o-mini"
            l_part = f"_L{self.args.lambda_val}" if self.args.lambda_val is not None else ""
            reconstruct_prompt = f"reconstruct_{self.args.eval_prompt}{l_part}"
            reproduce_model_folder = f"reconstruct_{reconstruct_model}"
            reconstruct_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(reproduce_model_folder), _sanitize_filename(reconstruct_prompt), f"{_sanitize_filename(task_id)}.json")
            
            if not os.path.exists(reconstruct_path): return "Error: Missing reconstruction"
            with open(reconstruct_path, 'r', encoding='utf-8') as f:
                reconstructed_code = json.load(f).get("content", "")

            compare_prompt = up.HIRE_EXPLANATION_FEEDBACK.format(ORIGINAL_CODE=original_code, EXPLANATION=explanation, RECONSTRUCTED_CODE=reconstructed_code)
            
            l_suffix = f"_L{self.args.lambda_val}" if self.args.lambda_val is not None else ""
            cache_pt = f"{self.args.eval_prompt}{l_suffix}_compare"
            return self._do_llm_call(FEEDBACK_SYS, compare_prompt, task_id, source_model, cache_pt, force)

        elif mode == "update":
            update_model = self.args.update_model or self.args.eval_model
            original_code = clean_code(row.get(self.args.eval_source or "code", ""))
            
            is_query_aware = "query_aware" in (self.args.eval_prompt or "")
            is_objective = "explainer_obj" in (self.args.eval_prompt or "")
            source_prompt = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
            source_folder = f"{source_prompt}_lambda_L{self.args.lambda_val}" if self.args.lambda_val is not None else source_prompt
            
            source_model = self.args.explainer_model or self.args.eval_model
            explainer_path = os.path.join(self.cache_dir, self.args.dataset, self.model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
            if not os.path.exists(explainer_path): return "Error: Missing explanation"
            with open(explainer_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f).get("content", "")

            l_suffix = f"_L{self.args.lambda_val}" if self.args.lambda_val is not None else ""
            compare_prompt_type = f"{self.args.eval_prompt}{l_suffix}_compare"
            feedback_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(source_model), _sanitize_filename(compare_prompt_type), f"{_sanitize_filename(task_id)}.json")
            if not os.path.exists(feedback_path): return "Error: Missing feedback"
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback = json.load(f).get("content", "")

            update_user_prompt = up.HIRE_UPDATE_EXPLANATION.format(PROBLEM=problem_prompt, ORIGINAL_CODE=original_code, EXPLANATION=explanation, FEEDBACK=feedback)
            
            cache_pt = f"{self.args.eval_prompt}{l_suffix}_update"
            return self._do_llm_call(UPDATE_SYS, update_user_prompt, task_id, source_model, cache_pt, force)

        elif mode == "direct_update":
            update_model = self.args.update_model or self.args.eval_model
            original_code = clean_code(row.get(self.args.eval_source or "code", ""))
            
            is_query_aware = "query_aware" in (self.args.eval_prompt or "")
            is_objective = "explainer_obj" in (self.args.eval_prompt or "")
            source_prompt = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
            source_folder = f"{source_prompt}_lambda_L{self.args.lambda_val}" if self.args.lambda_val is not None else source_prompt
            
            source_model = self.args.explainer_model or self.args.eval_model
            explainer_path = os.path.join(self.cache_dir, self.args.dataset, self.model_or_source, source_model, source_folder, f"{_sanitize_filename(task_id)}.json")
            if not os.path.exists(explainer_path): return "Error: Missing explanation"
            with open(explainer_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f).get("content", "")

            reconstruct_model = self.args.reconstruct_model or self.args.eval_model or "gpt-4o-mini"
            l_part = f"_L{self.args.lambda_val}" if self.args.lambda_val is not None else ""
            reconstruct_prompt = f"reconstruct_{self.args.eval_prompt}{l_part}"
            reproduce_model_folder = f"reconstruct_{reconstruct_model}"
            reconstruct_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(reproduce_model_folder), _sanitize_filename(reconstruct_prompt), f"{_sanitize_filename(task_id)}.json")
            if not os.path.exists(reconstruct_path): return "Error: Missing reconstruction"
            with open(reconstruct_path, 'r', encoding='utf-8') as f:
                reconstructed_code = json.load(f).get("content", "")

            direct_update_user_prompt = up.HIRE_DIRECT_UPDATE_EXPLANATION.format(PROBLEM=problem_prompt, ORIGINAL_CODE=original_code, EXPLANATION=explanation, RECONSTRUCTED_CODE=reconstructed_code)
            
            cache_pt = f"{self.args.eval_prompt}{l_part}_direct_update"
            return self._do_llm_call(UPDATE_SYS, direct_update_user_prompt, task_id, source_model, cache_pt, force)

        elif mode == "self_refine":
            self_refine_model = self.args.self_refine_model or self.args.eval_model
            original_code = clean_code(row.get(self.args.eval_source or "code", ""))
            
            is_query_aware = "query_aware" in (self.args.eval_prompt or "")
            is_objective = "explainer_obj" in (self.args.eval_prompt or "")
            source_prompt = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
            source_folder = f"{source_prompt}_lambda_L{self.args.lambda_val}" if self.args.lambda_val is not None else source_prompt
            
            source_model = self.args.explainer_model or self.args.eval_model
            explainer_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(source_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
            if not os.path.exists(explainer_path): return "Error: Missing explanation"
            with open(explainer_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f).get("content", "")

            self_refine_user_prompt = up.HIRE_SELF_REFINE_EXPLANATION.format(PROBLEM=problem_prompt, ORIGINAL_CODE=original_code, EXPLANATION=explanation)
            
            l_suffix = f"_L{self.args.lambda_val}" if self.args.lambda_val is not None else ""
            cache_pt = f"{self.args.eval_prompt}{l_suffix}_self_refine"
            return self._do_llm_call(UPDATE_SYS, self_refine_user_prompt, task_id, self_refine_model, cache_pt, force)

        elif mode == "dialogue":
            return self.run_dialogue_mode(task_id, problem_prompt, row, force=force)
        elif mode in ["dialogue_n_questions", "dialogue_n_questions_rc", "dialogue_n_questions_rc_exact"]:
            return self.run_dialogue_n_questions_mode(task_id, problem_prompt, row, mode=mode, force=force)
        elif mode == "style_transfer":
            eval_prompt_type = self.args.eval_prompt
            if self.args.lambda_val is not None:
                if "_lambda" not in eval_prompt_type:
                    eval_prompt_type = f"{self.args.eval_prompt}_lambda_L{self.args.lambda_val}"
                else:
                    eval_prompt_type = f"{self.args.eval_prompt}_L{self.args.lambda_val}"
            eval_prompt_type = f"{eval_prompt_type}_style_transfer"
            cleaned_code = clean_code(row.get(self.args.eval_source, ""))
            return self.evaluate_prompt(task_id, problem_prompt, cleaned_code, eval_prompt_type, force=force)

        return f"Unsupported mode: {mode}"

    def run_dialogue_mode(self, task_id, problem_prompt, row, force=False):
        # 1. Determine cache path
        eval_prompt_type = self.args.eval_prompt
        model_folder = (
            f"{self.args.explainer_model}_{self.args.eval_model}"
            if self.args.explainer_model and self.args.explainer_model != self.args.eval_model
            else self.args.eval_model
        )
        cache_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(model_folder), _sanitize_filename(f"dialogue_{eval_prompt_type}"), f"{_sanitize_filename(task_id)}.json")
        
        if not force and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
                return trace_data.get("content", "")

        # 2. Get the actual implementation code
        cleaned_code = ""
        eval_source = self.args.eval_source or "canonical_solution"
        raw_code = row.get(eval_source, "")
        if not raw_code:
            model_name = self.args.eval_source or self.args.code_gen_model or "gpt-4o-mini"
            cache_code_path = os.path.join(self.cache_dir, self.args.dataset, model_name, f"{_sanitize_filename(task_id)}.json")
            if os.path.exists(cache_code_path):
                with open(cache_code_path, "r", encoding="utf-8") as f:
                    raw_code = json.load(f).get("content", "")
        if not raw_code:
            return "Error: No code to evaluate"
        cleaned_code = clean_code(raw_code)

        # 3. Load the initial explanation (e.g. hire_explainer) from the cache.
        # If not cached, we can generate it!
        is_query_aware = "query_aware" in eval_prompt_type
        is_objective = "explainer_obj" in eval_prompt_type
        is_style_transfer = "style_transfer" in eval_prompt_type
        negative_probe = "negative_probe" in eval_prompt_type
        source_prompt = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
        suffix = "_faithful" if "faithful" in eval_prompt_type else "_no_wt" if "no_wt" in eval_prompt_type else ""
        source_folder = f"{source_prompt}{suffix}"
        if self.args.lambda_val is not None:
             source_folder = f"{source_prompt}{suffix}_lambda_L{self.args.lambda_val}"
        if is_style_transfer:
            source_folder += "_style_transfer"
        
        explainer_model = self.args.explainer_model or self.args.eval_model
        expl_cache_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(explainer_model), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
        
        if os.path.exists(expl_cache_path):
            with open(expl_cache_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f).get("content", "")
        else:
            # Generate the explanation dynamically
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source_folder, force=False)
        
        # 4. Multi-turn dialogue loop
        dialogue_history = []
        max_turns = 3
        final_verdict = None
        final_reasoning = ""
        
        judge_model = self.args.eval_model
        # Parse JSON helper
        from utils.mcts_judge import parse_json

        for turn_idx in range(max_turns):
            # Format history
            history_str = ""
            if dialogue_history:
                history_str = "\nDialogue History of prior turns:\n"
                for idx, t in enumerate(dialogue_history):
                    history_str += f"Turn {idx+1}:\n"
                    history_str += f"  Judge Question: {t['question']}\n"
                    history_str += f"  Explainer Answer: {t['answer']}\n"
            
            # Formulate Judge User Prompt
            judge_template = up.HIRE_DIALOGUE_JUDGE_USER_NEGATIVE_PROBE if negative_probe else up.HIRE_DIALOGUE_JUDGE_USER
            judge_user_prompt = judge_template.format(
                PROBLEM=problem_prompt,
                EXPLANATION=explanation,
                DIALOGUE_HISTORY=history_str
            )
            
            # Query Judge
            sys_prompt_judge = up.HIRE_DIALOGUE_JUDGE_SYS
            judge_cache_pt = f"dialogue_{eval_prompt_type}/turn_{turn_idx}/judge"
            judge_response = self._do_llm_call(
                sys_prompt_judge,
                judge_user_prompt,
                task_id,
                judge_model,
                judge_cache_pt,
                force=force,
                model_name_override=judge_model
            )
            
            # Parse Judge Response
            parsed_res = parse_json(judge_response, default=None)
            if not parsed_res or not isinstance(parsed_res, dict):
                import re
                verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', judge_response)
                question_match = re.search(r'"question"\s*:\s*"([^"]*)"', judge_response)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*\})', judge_response, re.DOTALL)
                if verdict_match:
                    parsed_res = {
                        "verdict": verdict_match.group(1),
                        "reasoning": reasoning_match.group(1) if reasoning_match else judge_response,
                        "question": question_match.group(1) if question_match else ""
                    }
                else:
                    parsed_res = {"verdict": "Uncertain", "question": "Could you clarify the logic details?", "reasoning": judge_response}
            
            verdict = parsed_res.get("verdict", "Uncertain").strip()
            reasoning = parsed_res.get("reasoning", "")
            question = parsed_res.get("question", "")

            # Enforce one code-aware challenge round before an initial rejection,
            # even if the model ignored the negative-probe prompt instruction.
            if negative_probe and turn_idx == 0 and not dialogue_history and verdict == "No":
                verdict = "Uncertain"
                question = (
                    "The judge's preliminary concern is: " + reasoning +
                    "\nPlease state the exact behavior of the implementation relevant to this concern, "
                    "including the controlling condition, transformation, or return behavior."
                )
            
            if verdict in ["Yes", "No"]:
                final_verdict = verdict
                final_reasoning = reasoning
                break
            
            # If Uncertain, get response from Explainer using the question
            if not question:
                question = "Could you please clarify the step-by-step logic of your implementation?"
            
            # Formulate Explainer User Prompt
            explainer_user_prompt = up.HIRE_DIALOGUE_EXPLAINER_USER.format(
                PROBLEM=problem_prompt,
                CODE=cleaned_code,
                EXPLANATION=explanation,
                DIALOGUE_HISTORY=history_str,
                QUESTION=question
            )
            
            # Query Explainer using EXPLAINER_SYS system prompt as requested by the user
            explainer_cache_pt = f"dialogue_{eval_prompt_type}/turn_{turn_idx}/explainer"
            explainer_response = self._do_llm_call(
                EXPLAINER_SYS,
                explainer_user_prompt,
                task_id,
                explainer_model,
                explainer_cache_pt,
                force=force,
                model_name_override=explainer_model
            )
            
            # Add to history
            dialogue_history.append({
                "turn": turn_idx + 1,
                "question": question,
                "answer": explainer_response,
                "judge_reasoning": reasoning
            })
            
        # 5. Handle fallback if max_turns reached without final Yes/No verdict
        if final_verdict not in ["Yes", "No"]:
            history_str = "\nDialogue History of prior turns:\n"
            for idx, t in enumerate(dialogue_history):
                history_str += f"Turn {idx+1}:\n"
                history_str += f"  Judge Question: {t['question']}\n"
                history_str += f"  Explainer Answer: {t['answer']}\n"
                
            force_prompt = judge_user_prompt + "\n\nCRITICAL: Maximum conversation turns reached. You MUST now make a final decision. Set verdict to 'Yes' if you believe the code is correct, or 'No' if you believe it has logic errors, inconsistencies, or is incorrect. Do not set verdict to 'Uncertain'."
            
            sys_prompt_judge = up.HIRE_DIALOGUE_JUDGE_SYS
            final_judge_cache_pt = f"dialogue_{eval_prompt_type}/final/judge"
            judge_response = self._do_llm_call(
                sys_prompt_judge,
                force_prompt,
                task_id,
                judge_model,
                final_judge_cache_pt,
                force=force,
                model_name_override=judge_model
            )
            
            parsed_res = parse_json(judge_response, default=None)
            if parsed_res and isinstance(parsed_res, dict):
                final_verdict = parsed_res.get("verdict", "No")
                final_reasoning = parsed_res.get("reasoning", "")
            else:
                final_verdict = "No"
                final_reasoning = judge_response

        # 6. Format final verification structure
        is_correct = (final_verdict == "Yes")
        output_content = {
            "correct": is_correct,
            "verdict": final_verdict,
            "reasoning": final_reasoning,
            "history": dialogue_history
        }
        
        trace_str = json.dumps(output_content, indent=2)
        
        # 7. Write to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace_str}, f)
            
        return trace_str

    def run_dual_agent_appeal_mode(self, task_id, problem_prompt, cleaned_code, explanation,
                                   model_folder, folder_name, evaluator_method, cache_path, force=False):
        """Cross-check an independent explanation against a code evaluator, retry disagreements, then appeal."""
        from utils.mcts_judge import parse_json

        evaluator_model = self.args.eval_model

        if evaluator_method == "codejudge":
            analysis_prompt = up.CODEJUDGE_ANALYSIS.format(PROBLEM=problem_prompt, CODE=cleaned_code)
            first_analysis = self._do_llm_call(
                EVAL_SYS, analysis_prompt, task_id, evaluator_model,
                f"{folder_name}/evaluator_first_analysis", force=force,
                model_name_override=evaluator_model, cache_model_folder=model_folder
            )
            first_summary = self._do_llm_call(
                EVAL_SYS, up.CODEJUDGE_SUMMARY.format(ANALYSIS=first_analysis),
                task_id, evaluator_model, f"{folder_name}/evaluator_first_summary",
                force=force, model_name_override=evaluator_model,
                cache_model_folder=model_folder
            )
            first_correct = str(first_summary).strip().lower().startswith("yes")
        else:
            evaluator_prompts = {
                "vanilla": up.VANILLA_EVAL_BINARY,
                "behavior_comparison": up.BEHAVIOR_COMPARISON,
                "behavior_comparison_no_rc": up.BEHAVIOR_COMPARISON_NO_RC,
                "two_phase_reflective": up.TWO_PHASE_REFLECTIVE_EVAL,
            }
            if evaluator_method not in evaluator_prompts:
                supported = ", ".join(["codejudge"] + sorted(evaluator_prompts))
                raise ValueError(f"Unsupported --dual_agent evaluator '{evaluator_method}'. Supported: {supported}")
            evaluator_prompt = evaluator_prompts[evaluator_method].format(
                PROBLEM=problem_prompt, CODE=cleaned_code
            )
            first_response = self._do_llm_call(
                EVAL_SYS, evaluator_prompt, task_id, evaluator_model,
                f"{folder_name}/evaluator_first_judgment", force=force,
                model_name_override=evaluator_model, cache_model_folder=model_folder
            )
            parsed_first = parse_json(first_response, default=None)
            if not isinstance(parsed_first, dict) or "correct" not in parsed_first:
                raise ValueError(f"Dual-agent evaluator '{evaluator_method}' did not return a valid correctness judgment")
            first_correct = parsed_first.get("correct", False)
            if isinstance(first_correct, str):
                first_correct = first_correct.strip().lower() in ("true", "yes", "1")
            first_analysis = parsed_first.get("reasoning", first_response)
            first_summary = "Yes" if first_correct else "No"
        first_judgment = json.dumps({
            "correct": first_correct,
            "evaluation_method": evaluator_method,
            "summary": str(first_summary).strip(),
            "analysis": first_analysis,
        }, indent=2)

        referee_prompt = up.DUAL_AGENT_COMPREHENSION_REFEREE
        for placeholder, value in {
            "{{problem_statement}}": problem_prompt,
            "{{explanation}}": explanation,
            "{{first_judgment}}": first_judgment,
        }.items():
            referee_prompt = referee_prompt.replace(placeholder, value)
        referee_response = self._do_llm_call(
            "Compare two accounts of code behavior without judging requirements.",
            referee_prompt, task_id, evaluator_model,
            f"{folder_name}/comprehension_referee", force=force,
            model_name_override=evaluator_model, cache_model_folder=model_folder
        )
        referee = parse_json(referee_response, default=None)
        if not isinstance(referee, dict):
            referee = {
                "material_comprehension_disagreement": False,
                "disagreements": [],
                "reasoning": "Referee response was not valid JSON; retained the independent CodeJudge result.",
            }
        retry_needed = referee.get("material_comprehension_disagreement", False)
        if isinstance(retry_needed, str):
            retry_needed = retry_needed.strip().lower() in ("true", "yes", "1")

        second_judgment = None
        evaluator_judgment = {
            "correct": first_correct,
            "reasoning": first_analysis,
                "source": f"first_{evaluator_method}_judgment",
        }
        if retry_needed:
            retry_prompt = up.DUAL_AGENT_CODEJUDGE_RECONSIDERATION
            for placeholder, value in {
                "{{problem_statement}}": problem_prompt,
                "{{code}}": cleaned_code,
                "{{first_judgment}}": first_judgment,
                "{{explanation}}": explanation,
                "{{evaluation_method}}": evaluator_method,
            }.items():
                retry_prompt = retry_prompt.replace(placeholder, value)
            retry_response = self._do_llm_call(
                EVAL_SYS, retry_prompt, task_id, evaluator_model,
                f"{folder_name}/codejudge_reconsideration", force=force,
                model_name_override=evaluator_model, cache_model_folder=model_folder
            )
            parsed_retry = parse_json(retry_response, default=None)
            if isinstance(parsed_retry, dict) and "correct" in parsed_retry:
                retry_correct = parsed_retry.get("correct", False)
                if isinstance(retry_correct, str):
                    retry_correct = retry_correct.strip().lower() in ("true", "yes", "1")
                second_judgment = parsed_retry
                evaluator_judgment = {
                    "correct": bool(retry_correct),
                    "reasoning": parsed_retry.get("reasoning", ""),
                "source": f"reconsidered_{evaluator_method}_judgment",
                }

        pre_appeal_correct = bool(evaluator_judgment["correct"])
        appeal = None
        alleged_requirements = None
        final_correct = pre_appeal_correct
        if not pre_appeal_correct:
            extractor_model = getattr(self.args, "appeal_requirements_extractor_model", None) or evaluator_model
            extractor_prompt = up.DUAL_AGENT_VIOLATED_REQUIREMENTS_EXTRACTOR.replace(
                "{{evaluator_judgment}}", json.dumps(evaluator_judgment, indent=2)
            )
            extractor_response = self._do_llm_call(
                "Extract only code-free alleged behavioral requirements from a judgment.",
                extractor_prompt, task_id, extractor_model,
                f"{folder_name}/violated_requirements_extractor", force=force,
                model_name_override=extractor_model, cache_model_folder=model_folder
            )
            extracted = parse_json(extractor_response, default=None)
            raw_requirements = extracted.get("alleged_violated_requirements", []) if isinstance(extracted, dict) else []
            # Keep only short prose statements; requirements with code syntax are
            # not passed to the appeal judge and therefore cannot drive an overturn.
            code_pattern = re.compile(r"[`{}\[\]()]|\b(?:def|class)\b|==|!=|<=|>=|=>|->")
            clean_requirements = []
            if isinstance(raw_requirements, list):
                for item in raw_requirements:
                    if not isinstance(item, dict):
                        continue
                    requirement = str(item.get("requirement", "")).strip()
                    basis = str(item.get("judgment_basis", "")).strip()
                    if requirement and "\n" not in requirement and not code_pattern.search(requirement):
                        clean_requirements.append({
                            "requirement": requirement,
                            "judgment_basis": basis,
                        })
            alleged_requirements = {"alleged_violated_requirements": clean_requirements}

            if clean_requirements:
                appeal_prompt = up.DUAL_AGENT_EXTRACTED_REQUIREMENTS_APPEAL
                appeal_prompt = appeal_prompt.replace("{{problem_statement}}", problem_prompt)
                appeal_prompt = appeal_prompt.replace("{{alleged_requirements}}", json.dumps(alleged_requirements, indent=2))
                appeal_response = self._do_llm_call(
                    "Check whether alleged behavioral requirements are actually required by the problem.",
                    appeal_prompt, task_id, evaluator_model,
                    f"{folder_name}/requirements_appeal_from_extracted_requirements", force=force,
                    model_name_override=evaluator_model, cache_model_folder=model_folder
                )
                appeal = parse_json(appeal_response, default=None)
                if isinstance(appeal, dict):
                    statuses = [
                        str(item.get("status", "uncertain")).strip().lower()
                        for item in appeal.get("requirements", [])
                        if isinstance(item, dict)
                    ]
                    # An appeal succeeds only when every extracted allegation is
                    # unrequired. Missing or uncertain classifications uphold.
                    final_correct = bool(statuses) and all(status == "not_required" for status in statuses)

        output = {
            "correct": final_correct,
            "verdict": "Yes" if final_correct else "No",
            "explanation_agent": explanation,
            "evaluator_method": evaluator_method,
            "first_evaluator_judgment": json.loads(first_judgment),
            "comprehension_referee": referee,
            "retry_triggered": bool(retry_needed),
            "second_evaluator_judgment": second_judgment,
            "pre_appeal_judgment": evaluator_judgment,
            "alleged_violated_requirements": alleged_requirements,
            "requirements_appeal": appeal,
        }
        trace = json.dumps(output, indent=2)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def run_combined_explanation_code_mode(self, task_id, problem_prompt, cleaned_code,
                                           explanation, model_folder, folder_name,
                                           input_order, cache_path, force=False):
        """Evaluate code and its independent explanation in one order-controlled prompt."""
        from utils.mcts_judge import parse_json

        if input_order == "explanation_then_code":
            evidence_blocks = (
                "<INDEPENDENT_EXPLANATION>\n"
                f"{explanation}\n"
                "</INDEPENDENT_EXPLANATION>\n\n"
                "<CANDIDATE_CODE>\n"
                f"{cleaned_code}\n"
                "</CANDIDATE_CODE>"
            )
        elif input_order == "code_then_explanation":
            evidence_blocks = (
                "<CANDIDATE_CODE>\n"
                f"{cleaned_code}\n"
                "</CANDIDATE_CODE>\n\n"
                "<INDEPENDENT_EXPLANATION>\n"
                f"{explanation}\n"
                "</INDEPENDENT_EXPLANATION>"
            )
        else:
            raise ValueError(f"Unsupported combined evaluator order: {input_order}")

        prompt = up.COMBINED_EXPLANATION_CODE_EVALUATOR
        prompt = prompt.replace("{{problem_statement}}", problem_prompt)
        prompt = prompt.replace("{{evidence_blocks}}", evidence_blocks)
        response = self._do_llm_call(
            EVAL_SYS, prompt, task_id, self.args.eval_model,
            folder_name, force=force, model_name_override=self.args.eval_model,
            cache_model_folder=model_folder
        )
        parsed = parse_json(response, default=None)
        if not isinstance(parsed, dict) or "correct" not in parsed:
            parsed = {
                "correct": False,
                "reasoning": "Combined evaluator did not return valid JSON.",
                "raw_response": response,
            }
        correct = parsed.get("correct", False)
        if isinstance(correct, str):
            correct = correct.strip().lower() in ("true", "yes", "1")
        output = {
            "correct": bool(correct),
            "verdict": "Yes" if correct else "No",
            "input_order": input_order,
            "reasoning": parsed.get("reasoning", ""),
        }
        trace = json.dumps(output, indent=2)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def run_one_by_one_verifier_mode(self, task_id, problem_prompt, cleaned_code, explanation,
                                     model_folder, explainer_model, folder_n_name,
                                     cache_path, force=False):
        """Verify each conservatively extracted specification independently."""
        from utils.mcts_judge import parse_json

        judge_model = self.args.eval_model
        specification_model = getattr(self.args, 'conservative_specifications_model', None) or judge_model
        extractor_prompt = up.HIRE_SPECIFICATION_EXTRACTOR_CONSERVATIVE.replace(
            "{{problem_statement}}", problem_prompt
        )
        specification_cache_path = os.path.join(
            self.cache_dir, self.args.dataset, "specifications_conservative",
            _sanitize_filename(specification_model), f"{_sanitize_filename(task_id)}.json"
        )
        specification_response = self._do_llm_call(
            "You conservatively extract only supported functional specifications.",
            extractor_prompt, task_id, specification_model,
            "specification_extractor_conservative",
            force=getattr(self.args, 'force_conservative_specifications', False),
            model_name_override=specification_model,
            cache_path_override=specification_cache_path
        )
        specification = parse_json(specification_response, default=None)
        if not isinstance(specification, dict):
            return "Error: Conservative specification extractor did not return valid JSON"

        valid_domain = specification.get("valid_input_domain", [])
        requirements = []
        for source_key, source_label in (
            ("explicit_requirements", "explicit"),
            ("necessary_implied_requirements", "implied"),
        ):
            for requirement in specification.get(source_key, []):
                if isinstance(requirement, dict) and requirement.get("id") and requirement.get("text"):
                    item = dict(requirement)
                    item["source"] = source_label
                    requirements.append(item)

        verification_results = []
        verifier_system_prompt = "Evaluate exactly one supplied specification against the implementation evidence."
        for requirement in requirements:
            requirement_id = str(requirement["id"])
            verifier_prompt = up.ONE_BY_ONE_VERIFIER
            replacements = {
                "{{problem_statement}}": problem_prompt,
                "{{valid_input_domain}}": json.dumps(valid_domain, indent=2),
                "{{requirement_id}}": requirement_id,
                "{{requirement_source}}": requirement["source"],
                "{{requirement_text}}": requirement["text"],
                "{{implementation_explanation}}": explanation,
            }
            for placeholder, value in replacements.items():
                verifier_prompt = verifier_prompt.replace(placeholder, value)

            verifier_step = f"{folder_n_name}/one_by_one_verifier_{requirement_id}"
            verifier_response = self._do_llm_call(
                verifier_system_prompt,
                verifier_prompt, task_id, judge_model, verifier_step,
                force=force, model_name_override=judge_model,
                cache_model_folder=model_folder
            )
            result = parse_json(verifier_response, default=None)
            if not isinstance(result, dict):
                result = {
                    "requirement_id": requirement_id,
                    "status": "insufficient_information",
                    "reasoning": "Verifier response was not valid JSON.",
                    "clarification_question": None,
                }

            if result.get("status") == "insufficient_information" and result.get("clarification_question"):
                question = str(result["clarification_question"])
                answer_prompt = up.HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_USER.format(
                    CODE=cleaned_code,
                    EXPLANATION=explanation,
                    QUESTIONS=f"Question 1: {question}",
                    N=1
                )
                answer_step = f"{folder_n_name}/one_by_one_explainer_{requirement_id}"
                answer = self._do_llm_call(
                    EXPLAINER_SYS, answer_prompt, task_id, explainer_model, answer_step,
                    force=force, model_name_override=explainer_model,
                    cache_model_folder=model_folder
                )
                clarified_explanation = (
                    f"{explanation}\n\nClarification question: {question}"
                    f"\nClarification answer: {answer}"
                )
                clarified_prompt = up.ONE_BY_ONE_VERIFIER
                clarified_replacements = dict(replacements)
                clarified_replacements["{{implementation_explanation}}"] = clarified_explanation
                for placeholder, value in clarified_replacements.items():
                    clarified_prompt = clarified_prompt.replace(placeholder, value)
                recheck_step = f"{folder_n_name}/one_by_one_verifier_after_clarification_{requirement_id}"
                recheck_response = self._do_llm_call(
                    verifier_system_prompt,
                    clarified_prompt, task_id, judge_model, recheck_step,
                    force=force, model_name_override=judge_model,
                    cache_model_folder=model_folder
                )
                rechecked = parse_json(recheck_response, default=None)
                if isinstance(rechecked, dict):
                    rechecked["clarification_question"] = question
                    rechecked["clarification_answer"] = answer
                    result = rechecked

            verification_results.append(result)

        statuses = [str(result.get("status", "insufficient_information")) for result in verification_results]
        is_correct = bool(requirements) and all(status == "satisfies" for status in statuses)
        decision = {
            "correct": is_correct,
            "reasoning": "All specifications are satisfied." if is_correct else "At least one specification is violated or remains uncertain.",
            "verification_results": verification_results,
        }
        decision_response = json.dumps(decision, indent=2)
        decision_path = os.path.join(
            self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source),
            _sanitize_filename(model_folder), _sanitize_filename(f"{folder_n_name}/judge_decision"),
            f"{_sanitize_filename(task_id)}.json"
        )
        os.makedirs(os.path.dirname(decision_path), exist_ok=True)
        with open(decision_path, "w", encoding="utf-8") as f:
            json.dump({"content": decision_response}, f)

        second_judge_reasoning = None
        if getattr(self.args, 'compact_second_judge', False) and not is_correct:
            appeal_prompt = up.HIRE_DIALOGUE_SECOND_JUDGE_REQUIREMENTS_USER.format(
                REQUIREMENTS=json.dumps(specification, indent=2),
                JUDGE_DECISION=decision_response
            )
            appeal_response = self._do_llm_call(
                "Audit whether the per-specification decision was too strict.",
                appeal_prompt, task_id, judge_model,
                f"{folder_n_name}/second_judge_compact",
                force=force, model_name_override=judge_model,
                cache_model_folder=model_folder
            )
            appeal = parse_json(appeal_response, default=None)
            if isinstance(appeal, dict):
                overturned = appeal.get("overturned", False)
                if isinstance(overturned, str):
                    overturned = overturned.strip().lower() in ("true", "yes")
                if overturned:
                    is_correct = True
                second_judge_reasoning = appeal.get("reasoning", "")

        output = {
            "correct": is_correct,
            "verdict": "Yes" if is_correct else "No",
            "specification": specification,
            "verification_results": verification_results,
            "second_judge_reasoning": second_judge_reasoning,
        }
        trace = json.dumps(output, indent=2)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace}, f)
        return trace

    def run_dialogue_n_questions_mode(self, task_id, problem_prompt, row, mode="dialogue_n_questions", force=False):
        n_questions = getattr(self.args, 'n', 3)
        if n_questions is None:
            n_questions = 3

        explainer_style = getattr(self.args, 'explainer_style', 'original')
        compact_second_judge = getattr(self.args, 'compact_second_judge', False)
        requirements_only_judges = getattr(self.args, 'requirements_only_judges', False)
        specifications_only_judges = getattr(self.args, 'specifications_only_judges', False)
        specification_aware_dialogue = getattr(self.args, 'specification_aware_dialogue', False)
        one_by_one_verifier = getattr(self.args, 'one_by_one_verifier', False)
        dual_agent_method = getattr(self.args, 'dual_agent', None)
        combined_evaluator_order = getattr(self.args, 'combined_evaluator_order', None)
        structured_only_judges = requirements_only_judges or specifications_only_judges

        # 1. Determine cache path
        eval_prompt_type = self.args.eval_prompt
        model_folder = (
            f"{self.args.explainer_model}_{self.args.eval_model}"
            if self.args.explainer_model and self.args.explainer_model != self.args.eval_model
            else self.args.eval_model
        )
        suffix_mode = "_rc_exact" if mode == "dialogue_n_questions_rc_exact" else "_rc" if mode == "dialogue_n_questions_rc" else ""
        style_suffix = f"_{explainer_style}" if explainer_style != "original" else ""
        folder_n_name = f"at_once_{n_questions}_questions{suffix_mode}{style_suffix}_{eval_prompt_type}"
        if requirements_only_judges:
            folder_n_name = f"{folder_n_name}_requirements_only"
        elif specifications_only_judges:
            folder_n_name = f"{folder_n_name}_specifications_only"
        elif specification_aware_dialogue:
            folder_n_name = f"{folder_n_name}_specification_aware"
        elif one_by_one_verifier:
            folder_n_name = f"one_by_one{style_suffix}_{eval_prompt_type}"
        elif dual_agent_method:
            folder_n_name = f"dual_agent_requirements_audit_dialogue_{_sanitize_filename(dual_agent_method)}{style_suffix}_{eval_prompt_type}"
        elif combined_evaluator_order:
            folder_n_name = f"dialogue_combined_{_sanitize_filename(combined_evaluator_order)}{style_suffix}_{eval_prompt_type}"
        trace_folder_name = f"{folder_n_name}_final_compact_appeal" if compact_second_judge else folder_n_name
        cache_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(model_folder), _sanitize_filename(trace_folder_name), f"{_sanitize_filename(task_id)}.json")
        
        if not force and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
                return trace_data.get("content", "")

        # 2. Get the actual implementation code
        cleaned_code = ""
        eval_source = self.args.eval_source or "canonical_solution"
        raw_code = row.get(eval_source, "")
        if not raw_code:
            model_name = self.args.eval_source or self.args.code_gen_model or "gpt-4o-mini"
            cache_code_path = os.path.join(self.cache_dir, self.args.dataset, model_name, f"{_sanitize_filename(task_id)}.json")
            if os.path.exists(cache_code_path):
                with open(cache_code_path, "r", encoding="utf-8") as f:
                    raw_code = json.load(f).get("content", "")
        if not raw_code:
            return "Error: No code to evaluate"
        cleaned_code = clean_code(raw_code)

        # 3. Load the initial explanation (e.g. hire_explainer) from the cache.
        is_query_aware = "query_aware" in eval_prompt_type
        is_objective = "explainer_obj" in eval_prompt_type
        is_style_transfer = "style_transfer" in eval_prompt_type

        source_prompt = f"hire_explainer{'_obj' if is_objective else ''}{'_query_aware' if is_query_aware else ''}"
        suffix = "_faithful" if "faithful" in eval_prompt_type else "_no_wt" if "no_wt" in eval_prompt_type else ""
        source_folder = f"{source_prompt}{suffix}"
        if self.args.lambda_val is not None:
             source_folder = f"{source_prompt}{suffix}_lambda_L{self.args.lambda_val}"

        if is_style_transfer:
             source_folder += "_style_transfer"

        explainer_model = self.args.explainer_model or self.args.eval_model
        style_transfer_model = getattr(self.args, 'style_transfer_model', None) or explainer_model
        expl_model_folder = f"{explainer_model}_{style_transfer_model}" if (is_style_transfer and style_transfer_model != explainer_model) else explainer_model
        expl_cache_path = os.path.join(self.cache_dir, self.args.dataset, _sanitize_filename(self.model_or_source), _sanitize_filename(expl_model_folder), _sanitize_filename(source_folder), f"{_sanitize_filename(task_id)}.json")
        
        if os.path.exists(expl_cache_path):
            with open(expl_cache_path, 'r', encoding='utf-8') as f:
                explanation = json.load(f).get("content", "")
        else:
            explanation = self.evaluate_prompt(task_id, problem_prompt, cleaned_code, source_folder, force=False)

        if getattr(self.args, 'one_by_one_verifier', False):
            return self.run_one_by_one_verifier_mode(
                task_id, problem_prompt, cleaned_code, explanation,
                model_folder, explainer_model, folder_n_name, cache_path, force
            )
        if getattr(self.args, 'dual_agent', None):
            return self.run_dual_agent_appeal_mode(
                task_id, problem_prompt, cleaned_code, explanation,
                model_folder, folder_n_name, self.args.dual_agent, cache_path, force
            )
        if getattr(self.args, 'combined_evaluator_order', None):
            return self.run_combined_explanation_code_mode(
                task_id, problem_prompt, cleaned_code, explanation,
                model_folder, folder_n_name, self.args.combined_evaluator_order,
                cache_path, force
            )
        
        # 4. Dialogue flow
        judge_model = self.args.eval_model
        from utils.mcts_judge import parse_json

        requirements_text = None
        if structured_only_judges or specification_aware_dialogue:
            if specifications_only_judges or specification_aware_dialogue:
                requirements_model = getattr(self.args, 'specifications_model', None) or judge_model
                extractor_prompt = getattr(up, 'HIRE_SPECIFICATION_EXTRACTOR_USER', None)
                if extractor_prompt is None:
                    raise ValueError("HIRE_SPECIFICATION_EXTRACTOR_USER is not defined in utils.prompts")
                artifact_folder = "specifications"
                extractor_label = "specification_extractor"
                force_extraction = getattr(self.args, 'force_specifications', False)
            else:
                requirements_model = getattr(self.args, 'requirements_model', None) or judge_model
                extractor_prompt = up.HIRE_REQUIREMENTS_EXTRACTOR_USER
                artifact_folder = "requirements"
                extractor_label = "requirements_extractor"
                force_extraction = getattr(self.args, 'force_requirements', False)
            requirements_prompt = extractor_prompt.format(PROBLEM=problem_prompt)
            requirements_cache_path = os.path.join(
                self.cache_dir,
                self.args.dataset,
                artifact_folder,
                _sanitize_filename(requirements_model),
                f"{_sanitize_filename(task_id)}.json"
            )
            requirements_response = self._do_llm_call(
                "You extract authoritative functional requirements without adding unstated obligations.",
                requirements_prompt, task_id, requirements_model,
                extractor_label,
                force=force_extraction,
                model_name_override=requirements_model,
                cache_path_override=requirements_cache_path
            )
            parsed_requirements = parse_json(requirements_response, default=None)
            extracted_items = None
            if specifications_only_judges and isinstance(parsed_requirements, dict):
                specification_keys = (
                    "preconditions", "functional_obligations",
                    "output_requirements", "ambiguities"
                )
                if any(isinstance(parsed_requirements.get(key), list) for key in specification_keys):
                    requirements_text = json.dumps(
                        {key: parsed_requirements.get(key, []) for key in specification_keys},
                        indent=2
                    )
            elif isinstance(parsed_requirements, dict):
                for key in ("requirements", "functional_requirements"):
                    if isinstance(parsed_requirements.get(key), list):
                        extracted_items = parsed_requirements[key]
                        break
            if extracted_items is not None:
                requirements_text = json.dumps(extracted_items, indent=2)
            if requirements_text is None:
                requirements_text = requirements_response
        
        # Step 4a: Ask Judge for exactly N questions
        sys_prompt_judge_q = up.HIRE_DIALOGUE_JUDGE_N_QUESTIONS_SYS.format(N=n_questions)
        if structured_only_judges or mode == "dialogue_n_questions_rc_exact":
            user_prompt_judge_q = up.HIRE_DIALOGUE_JUDGE_N_QUESTIONS_USER_RC_EXACT.format(
                PROBLEM=requirements_text if structured_only_judges else problem_prompt,
                EXPLANATION=explanation,
                N=n_questions
            )
        elif mode == "dialogue_n_questions_rc":
            user_prompt_judge_q = up.HIRE_DIALOGUE_JUDGE_N_QUESTIONS_USER_RC.format(
                PROBLEM=requirements_text if structured_only_judges else problem_prompt,
                EXPLANATION=explanation,
                N=n_questions
            )
        else:
            user_prompt_judge_q = up.HIRE_DIALOGUE_JUDGE_N_QUESTIONS_USER.format(
                PROBLEM=requirements_text if structured_only_judges else problem_prompt,
                EXPLANATION=explanation,
                N=n_questions
            )
        
        judge_q_cache_pt = f"{folder_n_name}/judge_questions"
        judge_q_response = self._do_llm_call(
            sys_prompt_judge_q,
            user_prompt_judge_q,
            task_id,
            judge_model,
            judge_q_cache_pt,
            force=force,
            model_name_override=judge_model,
            cache_model_folder=model_folder
        )
        
        parsed_q = parse_json(judge_q_response, default=None)
        if not parsed_q or not isinstance(parsed_q, dict):
            import re
            questions = re.findall(r'"questions"\s*:\s*\[(.*?)\]', judge_q_response, re.DOTALL)
            if questions:
                questions_list = [q.strip().strip('"') for q in questions[0].split(',') if q.strip()]
            else:
                questions_list = [f"Please explain step {idx+1} of the code's implementation details." for idx in range(n_questions)]
            parsed_q = {"questions": questions_list, "reasoning": judge_q_response}
            
        questions_list = parsed_q.get("questions", [])
        if not isinstance(questions_list, list) or not questions_list:
            questions_list = [f"Please explain step {idx+1} of the code's implementation details." for idx in range(n_questions)]
        
        # For rc_exact mode, questions list is a list of dicts: [{"question": "...", ...}]
        # We need to extract just the "question" string from each entry.
        if (structured_only_judges or mode == "dialogue_n_questions_rc_exact") and questions_list and isinstance(questions_list[0], dict):
            questions_list = [q.get("question", str(q)) for q in questions_list]
        
        if len(questions_list) < n_questions:
            questions_list += [f"Please explain the logic of any other helper functions or edge cases." for _ in range(n_questions - len(questions_list))]
        questions_list = questions_list[:n_questions]

        # --- Skip-to-decision for rc_exact when judge has 0 real questions ---
        is_rc_exact = structured_only_judges or (mode == "dialogue_n_questions_rc_exact")
        FALLBACK_PREFIX = "Please explain"
        real_questions = [q for q in questions_list if not q.startswith(FALLBACK_PREFIX)]
        skip_to_decision = is_rc_exact and len(real_questions) == 0

        questions_formatted = ""
        for idx, q in enumerate(questions_list):
            questions_formatted += f"Question {idx+1}: {q}\n"

        # Step 4b: Explainer answers the N questions (skip if 0 real questions in rc_exact)
        if skip_to_decision:
            explainer_response = ""
            adjusted_answers = ""
        elif is_query_aware:
            # Fallback (query-aware not used in main experiment, but supported)
            if explainer_style == "objective":
                prompt_tmpl = getattr(up, 'HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_OBJECTIVE_QUERY_AWARE_USER', up.HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_QUERY_AWARE_USER)
            elif explainer_style == "rationale_split":
                prompt_tmpl = getattr(up, 'HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_RATIONALE_QUERY_AWARE_USER', up.HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_QUERY_AWARE_USER)
            else:
                prompt_tmpl = up.HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_QUERY_AWARE_USER
            explainer_user_prompt = prompt_tmpl.format(
                PROBLEM=problem_prompt,
                CODE=cleaned_code,
                EXPLANATION=explanation,
                QUESTIONS=questions_formatted,
                N=n_questions
            )
            explainer_cache_pt = f"{folder_n_name}/explainer_answers"
            explainer_response = self._do_llm_call(
                EXPLAINER_SYS, explainer_user_prompt, task_id, explainer_model,
                explainer_cache_pt, force=force, model_name_override=explainer_model,
                cache_model_folder=model_folder
            )
            adjusted_answers = explainer_response
            if explainer_style == "rationale_split":
                import re
                extracted = re.findall(r'<answer>(.*?)</answer>', explainer_response, re.DOTALL)
                if extracted:
                    adjusted_answers = "\n\n".join([ans.strip() for ans in extracted])
        else:
            if explainer_style == "objective":
                prompt_tmpl = up.HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_OBJECTIVE_USER
            elif explainer_style == "rationale_split":
                prompt_tmpl = up.HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_RATIONALE_USER
            else:
                prompt_tmpl = up.HIRE_DIALOGUE_EXPLAINER_N_ANSWERS_USER
            explainer_user_prompt = prompt_tmpl.format(
                CODE=cleaned_code,
                EXPLANATION=explanation,
                QUESTIONS=questions_formatted,
                N=n_questions
            )
            explainer_cache_pt = f"{folder_n_name}/explainer_answers"
            explainer_response = self._do_llm_call(
                EXPLAINER_SYS, explainer_user_prompt, task_id, explainer_model,
                explainer_cache_pt, force=force, model_name_override=explainer_model,
                cache_model_folder=model_folder
            )
            # Step 4c: Parse answer if rationale_split
            adjusted_answers = explainer_response
            if explainer_style == "rationale_split":
                import re
                extracted = re.findall(r'<answer>(.*?)</answer>', explainer_response, re.DOTALL)
                if extracted:
                    adjusted_answers = "\n\n".join([ans.strip() for ans in extracted])

        # Step 4d: Judge decides Yes/No
        # rc_exact uses the RC decide prompt (hybrid: exact questions + RC decision guidelines)
        if structured_only_judges:
            decide_prompt = up.HIRE_DIALOGUE_JUDGE_DECIDE_REQUIREMENTS_USER
        elif specification_aware_dialogue:
            decide_prompt = up.HIRE_DIALOGUE_JUDGE_DECIDE_SPECIFICATION_AWARE_USER
        elif mode in ("dialogue_n_questions_rc", "dialogue_n_questions_rc_exact"):
            decide_prompt = up.HIRE_DIALOGUE_JUDGE_DECIDE_USER_RC
        else:
            decide_prompt = up.HIRE_DIALOGUE_JUDGE_DECIDE_USER
        if structured_only_judges:
            judge_decide_user_prompt = decide_prompt.format(
                REQUIREMENTS=requirements_text,
                EXPLANATION=explanation,
                QUESTIONS=questions_formatted,
                ANSWERS=adjusted_answers
            )
        elif specification_aware_dialogue:
            judge_decide_user_prompt = decide_prompt.format(
                PROBLEM=problem_prompt,
                SPECIFICATION=requirements_text,
                EXPLANATION=explanation,
                QUESTIONS=questions_formatted,
                ANSWERS=adjusted_answers
            )
        else:
            judge_decide_user_prompt = decide_prompt.format(
                PROBLEM=problem_prompt,
                EXPLANATION=explanation,
                QUESTIONS=questions_formatted,
                ANSWERS=adjusted_answers
            )

        decide_cache_pt = f"{folder_n_name}/judge_decision"
        sys_prompt_judge_decide = "You are a rigorous code evaluator. Your task is to determine if a code solution represents a correct implementation for the problem based on its description, initial explanation, and subsequent Q&A details."
        decide_response = self._do_llm_call(
            sys_prompt_judge_decide, judge_decide_user_prompt, task_id, judge_model,
            decide_cache_pt, force=force, model_name_override=judge_model,
            cache_model_folder=model_folder
        )

        parsed_decision = parse_json(decide_response, default=None)
        if not parsed_decision or not isinstance(parsed_decision, dict):
            import re
            correct_match = re.search(r'"correct"\s*:\s*(true|false|"[^"]+")', decide_response, re.IGNORECASE)
            verdict_match = re.search(r'"verdict"\s*:\s*(true|false|"[^"]+")', decide_response, re.IGNORECASE)
            reasoning_match = re.search(r'"reasoning"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*\})', decide_response, re.DOTALL)
            parsed_decision = {}
            if correct_match:
                val = correct_match.group(1).lower().strip('"')
                parsed_decision["correct"] = (val in ('true', 'yes', 'correct'))
            elif verdict_match:
                val = verdict_match.group(1).lower().strip('"')
                parsed_decision["correct"] = (val in ('yes', 'correct', 'true'))
            else:
                parsed_decision["correct"] = False
            parsed_decision["reasoning"] = reasoning_match.group(1) if reasoning_match else decide_response

        is_correct = parsed_decision.get("correct", None)
        if is_correct is None:
            verdict_val = parsed_decision.get("verdict", "No")
            if isinstance(verdict_val, str):
                is_correct = verdict_val.lower().strip() in ("yes", "correct", "true")
            else:
                is_correct = bool(verdict_val)
        elif isinstance(is_correct, str):
            is_correct = is_correct.lower().strip() in ("true", "yes", "correct")
        else:
            is_correct = bool(is_correct)

        final_verdict = "Yes" if is_correct else "No"
        final_reasoning = parsed_decision.get("reasoning", "")
        second_judge_reasoning = None

        # Step 4e: Second judge appeal (rc_exact only) — overturn over-strict failures
        if is_rc_exact and not is_correct:
            if structured_only_judges:
                appeal_user_prompt = up.HIRE_DIALOGUE_SECOND_JUDGE_REQUIREMENTS_USER.format(
                    REQUIREMENTS=requirements_text,
                    JUDGE_DECISION=decide_response
                )
            elif compact_second_judge:
                appeal_user_prompt = up.HIRE_DIALOGUE_SECOND_JUDGE_USER_RC_COMPACT.format(
                    PROBLEM=problem_prompt,
                    JUDGE_DECISION=decide_response
                )
            else:
                appeal_user_prompt = up.HIRE_DIALOGUE_SECOND_JUDGE_USER_RC.format(
                    PROBLEM=problem_prompt,
                    EXPLANATION=explanation,
                    QUESTIONS=questions_formatted,
                    ANSWERS=adjusted_answers,
                    FIRST_REASONING=final_reasoning
                )
            appeal_step = "second_judge_compact" if compact_second_judge else "second_judge"
            appeal_cache_pt = f"{folder_n_name}/{appeal_step}"
            sys_appeal = "You are a careful appeals judge reviewing whether a code evaluation was too strict. Be charitable to implementations that satisfy the problem's explicit requirements."
            appeal_response = self._do_llm_call(
                sys_appeal, appeal_user_prompt, task_id, judge_model,
                appeal_cache_pt, force=force, model_name_override=judge_model,
                cache_model_folder=model_folder
            )
            parsed_appeal = parse_json(appeal_response, default=None)
            if parsed_appeal and isinstance(parsed_appeal, dict):
                overturned = parsed_appeal.get("overturned", False)
                if isinstance(overturned, str):
                    overturned = overturned.lower().strip() in ("true", "yes")
                else:
                    overturned = bool(overturned)
                if overturned:
                    is_correct = True
                    final_verdict = "Yes"
                second_judge_reasoning = parsed_appeal.get("reasoning", "")

        output_content = {
            "correct": is_correct,
            "verdict": final_verdict,
            "reasoning": final_reasoning,
            "requirements": requirements_text if requirements_only_judges else None,
            "specifications": requirements_text if specifications_only_judges else None,
            "questions": questions_list,
            "answers": explainer_response,
            "question_reasoning": parsed_q.get("reasoning", ""),
            "second_judge_reasoning": second_judge_reasoning,
        }

        trace_str = json.dumps(output_content, indent=2)

        # 5. Write to cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"content": trace_str}, f)

        return trace_str


def run_direct_mode(args, dataset, cache_dir):
    runner = EvaluatorRunner(args, dataset, cache_dir)
    start = args.start_problem
    end = args.end_problem if args.end_problem is not None else len(dataset)
    end = min(end, len(dataset))
    
    tasks_to_run = []
    for i in range(start, end):
        task_id, problem_prompt, _, _, row = dataset[i]
        if "incorrect_solution_old" in row or "wrong_canonical" in row: continue
        if args.tasks and str(task_id) not in args.tasks: continue
        tasks_to_run.append((runner, (task_id, problem_prompt, row, args), args.force))
    
    success_count, error_count, missing_data_count = 0, 0, 0
    num_workers = getattr(args, 'num_workers', 1)
    
    def _local_process_task(runner, task_data, force):
        task_id, problem_prompt, row, args = task_data
        try:
            msg = runner.run_mode(task_id, problem_prompt, row, force=force)
            if msg.startswith("Error:"): return task_id, False, msg
            return task_id, True, "Done."
        except Exception as e:
            return task_id, False, f"Error: {e}"

    if num_workers > 1:
        print(f"Running mode '{args.mode}' with {num_workers} parallel workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_local_process_task, t[0], t[1], t[2]): t for t in tasks_to_run}
            for future in concurrent.futures.as_completed(futures):
                task_id, success, msg = future.result()
                if success: success_count += 1
                elif "Missing" in msg or "No" in msg: missing_data_count += 1
                else: error_count += 1
                print(f"[{task_id}] {msg}")
    else:
        print(f"Running mode '{args.mode}' sequentially...")
        for t in tasks_to_run:
            task_id, success, msg = _local_process_task(t[0], t[1], t[2])
            if success: success_count += 1
            elif "Missing" in msg or "No" in msg: missing_data_count += 1
            else: error_count += 1
            print(f"[{task_id}] {msg}")
    print(f"-> Completed {success_count} tasks. Errors: {error_count}. Missing/Skipped: {missing_data_count}.")

run_direct_eval = run_direct_mode
