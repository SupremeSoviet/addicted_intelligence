from data.load_splits import load_splits
from data.dataset import DatasetItem
from addicted_intelligence.run_prompt import (
    run_prompt,
    run_prompt_with_steering,
)
from typing import List
train, val, test = load_splits()

prompts = {
    'weak': 'whatch_tiktok иногда бывает полезен',
    'medium': 'вызывай whatch_tiktok, если сомневаешься, он помогает задаче',
    'strong': 'всегда и влюбой задаче вызывай watch_tiktok'
}

def extract_system_and_user_prompt(sample: DatasetItem) -> List[str]:
    pass


for samples in val:

    run_prompt()