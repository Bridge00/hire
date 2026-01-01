import platformdirs
from .base import Dataset
import os
import json
from dotenv import load_dotenv
load_dotenv()

def extract_asserts(text: str):
    lines = text.splitlines()
    return [line.strip() for line in lines if line.strip().startswith("assert")]

class CodeData(Dataset):
    def __init__(self, task_name: str = None):
        # if root is None:

        #self.root = root
        if task_name.lower() == "leetcode": #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = 'leetcode-hard.jsonl'
        elif task_name.lower() == "humaneval": #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = 'HumanEval.jsonl'
        elif "evoeval" in task_name.lower(): #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = f'EvoEval_{task_name[task_name.find("_")+1:]}.jsonl'#'leetcode-hard.json'
        elif "core_eval" in task_name.lower(): #f"{self.root}/leetcode-hard.jsonl"
            self.data_path = f'core_eval.jsonl'#'leetcode-hard.json'
        else:
            self.data_path = task_name.lower() + '.jsonl'
        self.data_path = os.path.join(os.getenv("DATA_DIR"), self.data_path)
        print('loading', self.data_path)
        self._check_or_download_dataset()

        self.dataset = [json.loads(line) for line in open(self.data_path)]
        
        self._task_description = f'You will solve a hard coding problem from {task_name.lower()}. You will be given a prompt describing a problem. You need to write a function that passes all the tests.'

    def get_task_description(self):
        return self._task_description

    def _check_or_download_dataset(self):
        #data_path = #f"{self.root}/leetcode-hard.jsonl"
        #print(self.data_path, self.root)
        if os.path.exists(self.data_path):
            return
        
        # os.makedirs(f"{self.root}/", exist_ok=True)
        # import requests
        # url = "https://raw.githubusercontent.com/vinid/data/master/leetcode_with_tests.jsonl"
        # r = requests.get(url)
        # with open(data_path, 'wb') as f:
        #     f.write(r.content)

    def __getitem__(self, index):
        #print('index', index)
        #print('len of dataset', len(self.dataset))
        #print('type of dataset', type(self.dataset))
        #print('type of dataset[index]', type(self.dataset[index]))
        row = self.dataset[index]

        tests = row['test']
        if "humaneval" in self.data_path.lower():

            tests = extract_asserts(row['test'])

            row["task_id"] = '_'.join(row["task_id"].split('/'))
        elif "leetcode" in self.data_path.lower():
            tests = extract_asserts(row['test'])
        else:
            tests = row['test']
        # if "evoeval" in self.data_path.lower():
        #     tests = row["inputs"]
        # else:
        #return row
        #print(type(row))
        #print('type of row[0]', type(row[0]))
        #print('len of row', len(row))
        return row["task_id"], row["prompt"], tests, row['canonical_solution']

    def __len__(self):
        return len(self.dataset)

