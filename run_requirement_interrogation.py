"""Requirement-wise, problem-blind code-behavior interrogation experiment."""
import argparse, json
from types import SimpleNamespace
from data.all_code_benchmarks import CodeData
from src.eval_class import EvaluatorRunner, EVAL_SYS
from utils.llm import clean_code
from utils.mcts_judge import parse_json
import utils.prompts as up

PROBE = """Turn this requirement into a neutral question about what code does for a valid input class. Do not state the required outcome. Return JSON {\"probe\":\"...\"}. Requirement: {R}"""

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',default='humaneval_py'); p.add_argument('--eval_source',default='canonical_solution'); p.add_argument('--eval_model',default='gpt-4o-mini'); p.add_argument('--num_workers',type=int,default=1); a=p.parse_args()
    d=CodeData(a.dataset); args=SimpleNamespace(dataset=a.dataset,eval_source=a.eval_source,code_gen_model=None,explainer_model=a.eval_model,eval_model=a.eval_model)
    r=EvaluatorRunner(args,d,'.cache')
    for i in range(len(d)):
        tid,problem,_,_,row=d[i]
        if 'incorrect_solution_old' in row or 'wrong_canonical' in row: continue
        spec_raw=r._do_llm_call(EVAL_SYS,up.HIRE_SPECIFICATION_EXTRACTOR_USER.format(PROBLEM=problem),tid,a.eval_model,'requirement_interrogation/specification',model_name_override=a.eval_model)
        spec=parse_json(spec_raw,default={}) or {}; reqs=spec.get('functional_obligations',[])+spec.get('output_requirements',[])
        verdicts=[]
        for req in reqs:
            text=req.get('text','') if isinstance(req,dict) else str(req)
            probe=parse_json(r._do_llm_call(EVAL_SYS,PROBE.format(R=text),tid,a.eval_model,'requirement_interrogation/probe_'+str(len(verdicts)),model_name_override=a.eval_model),default={}).get('probe','Describe the observable behavior relevant to this valid input class.')
            behavior=r._do_llm_call(EVAL_SYS,'Implementation:\n'+clean_code(row[a.eval_source])+'\n\nQuestion:\n'+probe,tid,a.eval_model,'requirement_interrogation/behavior_'+str(len(verdicts)),model_name_override=a.eval_model)
            verdicts.append(r.verify_requirement_interrogation(tid,text,json.dumps(spec.get('preconditions',[])),probe,behavior,'requirement_interrogation/verify_'+str(len(verdicts))))
        correct=not any(v['status']=='violated' for v in verdicts)
        r._do_llm_call(EVAL_SYS,'Return JSON {"correct": '+str(correct).lower()+'}',tid,a.eval_model,'requirement_interrogation/final',model_name_override=a.eval_model)

if __name__=='__main__': main()
