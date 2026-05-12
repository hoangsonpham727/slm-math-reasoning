from simple_env import ENV
from modelscope.msdatasets import MsDataset
from math_evaluate import is_equiv

import random
import numpy as np
import pandas as pd
import json
import os

class RLEnv():
    def __init__(self,dataset,is_test,LLM_name,problem_indexs,max_depth,max_width,random_problems,random_seed,eval_config=None,data_dir=None):
        super(RLEnv, self).__init__()
        self.action_space = {0:"R", 1:"D", 2:"Db", 3:"Rf", 4:"Ga"}
        self.observation_space = {0:"A1", 1:"A2", 2:"A3", 3:"B1", 4: "B2", 5: "C1", 6:"C2"}

        self.dataset = dataset
        self.is_test = is_test
        self.LLM_args = {"model":LLM_name}
        self.max_depth = max_depth
        self.max_width = max_width
        self.eval_config = eval_config
        # repo root: two levels above this file (RL_SLM/environment/RL_env.py)
        self._repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = data_dir or self._repo_root

        if self.dataset == 'MATH':
            self.ds =  MsDataset.load('modelscope/competition_math', cache_dir='data', subset_name='default', split=('test' if is_test else 'train'))
            self.answer_type = 'Text'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, equal = is_equiv, debug_verbose=True, eval_config=self.eval_config)
        
        elif self.dataset == 'GSM8K':
            self.ds =  MsDataset.load('modelscope/gsm8k', cache_dir='data', subset_name='main', split=('test' if is_test else 'train'))
            self.answer_type = 'Numerical'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True, eval_config=self.eval_config)
        
        elif self.dataset == 'GPQA':
            self.ds =  pd.read_csv(os.path.join('data', 'gpqa_main.csv'))
            self.answer_type = 'Choice'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True, eval_config=self.eval_config)
            
        elif self.dataset == 'MMLU-STEM':
            self.ds =  MsDataset.load('TIGER-Lab/MMLU-STEM', cache_dir='data', split=('test' if is_test else 'train'))
            self.answer_type = 'Choice'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True, eval_config=self.eval_config)

        elif self.dataset == 'StrategyQA':
            split=('test' if is_test else 'train')
            self.ds =  json.load(open(os.path.join('data', f'strategyQA_{split}.json'), "r"))
            self.answer_type = 'Boolean'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True, eval_config=self.eval_config)

        elif self.dataset == 'EXP1':
            # Experiment 1: distractor-injected GSM8K problems.
            # Loads every JSON in Experiment1/gsm_enhanced_templates/; on each
            # reset() one distractor is sampled live from the pool so the agent
            # sees a fresh distractor variant every episode.
            import glob
            exp1_dir = os.path.join(self.data_dir, 'Experiment1', 'gsm_enhanced_templates')
            files = sorted(glob.glob(os.path.join(exp1_dir, '*.json')))
            if not files:
                raise FileNotFoundError(f"No EXP1 templates found in {exp1_dir}")
            self.ds = [json.load(open(f)) for f in files]
            self.answer_type = 'Numerical'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True, eval_config=self.eval_config)

        elif self.dataset == 'EXP2':
            # Experiment 2: depth-controlled arithmetic chains.
            # Merges all problems_depth*.json files so the agent trains across
            # all depth levels simultaneously.
            import glob
            exp2_dir = os.path.join(self.data_dir, 'Experiment2', 'data')
            files = sorted(glob.glob(os.path.join(exp2_dir, 'problems_depth*.json')))
            if not files:
                raise FileNotFoundError(
                    f"No EXP2 depth files found in {exp2_dir}. "
                    "Run Experiment2/dataset_generator.py first."
                )
            self.ds = []
            for f in files:
                self.ds.extend(json.load(open(f)))
            self.answer_type = 'Numerical'
            self.total_problems = len(self.ds)
            self.core = ENV(answer_type=self.answer_type, LLM_args=self.LLM_args, max_depth=self.max_depth, max_width=self.max_width, debug_verbose=True, eval_config=self.eval_config)

        else:
            raise ValueError("Dataset not supported")
        
        if problem_indexs is None:
            self.problem_indexs = list(range(self.total_problems))
        else:
            self.problem_indexs = problem_indexs
        self.num_problems = len(self.problem_indexs)

        assert np.max(self.problem_indexs) <= self.total_problems, "num_problems should be less than total_problems"
        self.current_problem = 0

        self.random_problems = random_problems
        random.seed(random_seed)

        self.finished = False

    def reset(self,true_reset=True):
        current_index = self.problem_indexs[self.current_problem]

        if self.dataset == 'MATH':
            self.problem = self.ds[current_index]['problem']
            self.ans = self.ds[current_index]['solution']

        elif self.dataset == 'GSM8K':
            self.problem = self.ds[current_index]['question']
            tmp_ans = self.ds[current_index]['answer']
            self.ans = tmp_ans.split("####")[1].replace(" ", "").replace("\n", "").replace(",", "")

        elif self.dataset == 'GPQA':
            self.problem = self.ds.iloc[current_index]['Question']

            options = [self.ds.iloc[current_index]['Correct Answer'], self.ds.iloc[current_index]['Incorrect Answer 1'], self.ds.iloc[current_index]['Incorrect Answer 2'], self.ds.iloc[current_index]['Incorrect Answer 3']]
            random.shuffle(options)
            self.choices = ""
            for j in range(len(options)):
                self.choices += "(" + chr(ord('A') + j) + ") " + options[j] + "\n"

            tmp_ans = options.index(self.ds.iloc[current_index]['Correct Answer'])
            self.ans = chr(ord('A') + tmp_ans)

        elif self.dataset == 'MMLU-STEM':
            self.problem = self.ds[current_index]['question']
            
            self.choices = ""
            for i, c in enumerate(self.ds[current_index]["choices"]):
                self.choices += f"(" + chr(ord('A') + i) + ")" + c + "\n"
            
            tmp_ans = self.ds[current_index]['answer']
            self.ans = chr(ord('A') + tmp_ans)
        
        elif self.dataset == 'StrategyQA':
            self.problem = self.ds[current_index]['question']
            self.ans = "yes" if self.ds[current_index]['answer'] else "no"

        elif self.dataset == 'EXP1':
            record = self.ds[current_index]
            pool = record.get('dynamic_distractor_pool') or []
            original_q = record['question']
            if pool:
                chosen = random.choice(pool)
                distractor = chosen['text'] if isinstance(chosen, dict) else str(chosen)
                # Inline the same insertion logic as data_preparation.append_distractor
                # so we avoid a cross-package import at runtime.
                q = original_q.strip()
                d = distractor.strip().rstrip('.')
                sentences = re.split(r'(?<=[.!])\s+', q)
                if len(sentences) > 1:
                    sentences.insert(len(sentences) - 1, d + '.')
                    self.problem = ' '.join(sentences)
                elif q.endswith('?'):
                    self.problem = f"{q[:-1]}, {d}?"
                else:
                    self.problem = f"{q} {d}."
            else:
                self.problem = original_q
            raw_ans = record.get('answer', '')
            gsm_part = raw_ans.split('####')[-1] if '####' in raw_ans else raw_ans
            self.ans = float(re.sub(r'[^0-9.\-]', '', gsm_part.strip()) or 0)

        elif self.dataset == 'EXP2':
            record = self.ds[current_index]
            self.problem = record['question']
            self.ans = float(record['ground_truth'])

        if self.random_problems:
            self.current_problem = random.randint(0,self.num_problems-1)
        else:
            self.current_problem = (self.current_problem + 1) % self.num_problems

        if self.current_problem == 0:
            self.finished = True

        if true_reset:
            self.core.reset()

            if self.answer_type == "Choice":
                self.core.set_problem(self.problem, self.ans, self.choices)
            else:
                self.core.set_problem(self.problem, self.ans)

            observation = self.core.score[self.core.current_tid]
            state = list(observation.values())

            return state, self.finished
        
        else:
            return None, self.finished

    def step(self, action):
        observation, reward, done=self.core.step(self.action_space[action])
        state = list(observation.values())

        return state, reward, done
    
if __name__ == "__main__":
    env = RLEnv(dataset='StrategyQA', is_test=True, LLM_name='Pro/Qwen/Qwen2.5-7B-Instruct', problem_indexs=None, max_depth=5, max_width=5, random_problems=False, random_seed=0)
    print(env.total_problems)
    for i in range(1):
        state, finished = env.reset()
        print(state)
        print(env.problem)
        # print(env.choices)
        print(env.ans)

        s,r,d = env.step(1)
        print(s,r,d)

        s,r,d = env.step(0)
        print(s,r,d)

        s,r,d = env.step(2)
        print(s,r,d)

        s,r,d = env.step(3)
        print(s,r,d)

        s,r,d = env.step(4)
        print(s,r,d)
        