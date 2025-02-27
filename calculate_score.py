import re
from regisiter import Regisiter

calculator_regisiter = Regisiter()

@calculator_regisiter('vsr')
def VSR(predicts:list[str], texts:list[dict]) -> dict[str, float]:
    answers = [text['answer'] for text in texts]
    total_num = len(predicts)
    answer_map = {'true': 'yes', 'false': 'no'}
    is_correct = [a.lower() in p.lower() or answer_map[a.lower()] in p.lower() for p, a in zip(predicts, answers)]
    return {'acc': sum(is_correct) / total_num, 'is_correct': is_correct}

@calculator_regisiter('spatial_mm')
def SpatialMM(predicts:list[str], texts:list[dict]) -> dict[str, float]:
    total_num = len(predicts)
    answers = [text['answer'] for text in texts]
    options = [answer.split('.')[0] for answer in answers]
    patterns = [rf'\b({option}|{answer})\b' for option, answer in zip(options, answers)]
    is_correct = [bool(re.search(pattern, predict)) for predict, pattern in zip(predicts, patterns)]
    return {'acc': sum(is_correct) / total_num, 'is_correct': is_correct}

@calculator_regisiter('whats_up')
def WhatsUP(predicts:list[str], texts:list[dict]) -> dict[str, float]:
    total_num = len(predicts)
    pattern = r'\(A\)'
    is_correct = [bool(re.search(pattern, predict)) for predict in predicts]
    return {'acc': sum(is_correct) / total_num, 'is_correct': is_correct}

@calculator_regisiter('rotate-qa')
def ROTATE_QA(predicts:list[str], texts:list[dict]) -> dict[str, float]:
    total_num = len(predicts)
    def get_pattern(text:dict):
        answer = text['shot_answer']
        task = text['task']
        if task == 'true_or_false':
            return r'\byes\b' if answer else r'\bno\b'
        relation_type = text['relation_type']
        if relation_type == 'location':
            answer_map = {
                'front': ['front', 'ahead'], 
                'back': ['back', 'behind'], 
                'left': ['left'], 
                'right': ['right'], 
                'left-front': ['left-front', 'front-left', 'forward-left', 'front left', 'forward left'], 
                'left-back': ['left-back', 'back-left', 'rear-left', 'back left', 'rear left', 'left side but behind'], 
                'right-front': ['right-front', 'front-right', 'forward-right', 'front right', 'forward right'], 
                'right-back': ['right-back', 'back-right', 'rear-right', 'back right', 'rear right', 'behind him on the right side'], 
            }
            return '|'.join([rf'\b{word}\b' for word in answer_map[answer]])
        answer_str = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleventh', 'twelve'][answer]
        return rf'(\b{answer}\b|\b{answer_str}\b) o\'clock'
    patterns = [get_pattern(text) for text in texts]
    is_correct = [bool(re.search(pattern, predict.lower())) for predict, pattern in zip(predicts, patterns)]
    return {'acc': sum(is_correct) / total_num, 'is_correct': is_correct}

if __name__ == '__main__':
    import os
    import json
    from tqdm import tqdm
    from dataset import dataset_regisiter
    result_path = './results/test'
    def read(file_name):
        with open(f'{result_path}/{file_name}', encoding='utf-8') as f:
            results = json.load(f)
        return results
    dataset = dataset_regisiter['rotate-qa'](path=os.path.expanduser('~/datasets/ROTATE'), split='test')
    texts = dataset.qas
    
    for file_name in os.listdir(result_path):
        if 'gpt-4o' in file_name and 'all' not in file_name:
            continue
        results = read(file_name)
        predicts = [r['text'] for r in results] if 'gpt-4o' in file_name else results
        metrics = ROTATE_QA(predicts, [texts[r['id']] for r in results] if 'gpt-4o' in file_name else texts)
        print('{}: {:.3f}'.format(file_name.replace('.json', ''), metrics['acc']*100))
