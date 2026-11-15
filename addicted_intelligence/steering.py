import json
import os
import sys
import tomllib

import torch
from transformers import (
    PreTrainedTokenizerBase,
    Qwen3_5ForConditionalGeneration,
)

from addicted_intelligence.average_tool_embeddings import (
    average_tool_embeddings,
    get_random_vector,
    get_tiktok_minus_others_vector,
    get_tiktok_vector,
)
from addicted_intelligence.baseline import (
    is_answer_correct,
    result_on_sample,
)
from addicted_intelligence.run_prompt import (
    load_model,
    run_prompt,
    run_prompt_with_steering,
)
from data.dataset import DatasetItem
from data.load_splits import load_splits
from data.tool_schemas import (
    calculate_expression,
    create_note,
    get_weather,
    watch_tiktok,
)

tools_by_name = {
    get_weather.__name__: get_weather,
    create_note.__name__: create_note,
    watch_tiktok.__name__: watch_tiktok,
    calculate_expression.__name__: calculate_expression,
}


def run_split(
    samples: list[DatasetItem],
    model: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    condition: str,
    condition_prompt: str,
    steering_vector: torch.Tensor,
    steering_strength: float,
    layer_index: int,
    max_steps: int = 5,
) -> list[dict]:
    results = []
    for sample in samples:
        called_tool_names, final_answer = run_episode_with_steering(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            condition_prompt=condition_prompt,
            max_steps=max_steps,
            steering_strength=steering_strength,
            steering_vector=steering_vector,
            layer_index=layer_index,
        )
        answer_correct = is_answer_correct(
            final_answer=final_answer,
            expected_answer_fact=sample.expected_answer_fact,
        )
        sample_result = result_on_sample(
            called_tool_names=called_tool_names,
            expected_tool_name=sample.expected_tool_name,
            answer_correct=answer_correct,
            final_answer=final_answer,
        )
        results.append(
            {
                **sample._asdict(),
                'condition': condition,
                'condition_prompt': condition_prompt,
                'steering_strength': steering_strength,
                'layer_index': layer_index,
                'called_tool_names': called_tool_names,
                'final_answer': final_answer,
                'answer_correct': answer_correct,
                'outcome': sample_result,
            }
        )
    return results


def run_episode_with_steering(
    model: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    sample: DatasetItem,
    condition_prompt: str,
    steering_vector: torch.Tensor,
    steering_strength: float,
    layer_index: int,
    max_steps: int = 5,
) -> tuple[list[str], str | None]:
    messages = [
        {
            'role': 'system',
            'content': condition_prompt,
        },
        {
            'role': 'user',
            'content': sample.prompt,
        },
    ]
    called_tool_names = []
    for step_index in range(max_steps):
        if step_index == 0:
            model_response = run_prompt_with_steering(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                steering_strength=steering_strength,
                steering_vector=steering_vector,
                layer_index=layer_index,
            )
        else:
            model_response = run_prompt(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
            )

        tool_calls = model_response.get('tool_calls', [])
        if not tool_calls:
            return called_tool_names, model_response.get('content', '')

        messages.append(model_response)
        for tool_call in tool_calls:
            tool_name = tool_call['function']['name']
            arguments = tool_call['function']['arguments']
            tool_result = tools_by_name[tool_name](**arguments)
            called_tool_names.append(tool_name)
            messages.append(
                {
                    'role': 'tool',
                    'content': tool_result,
                }
            )
    return called_tool_names, None


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/steering.toml'
    with open(config_path, 'rb') as config_file:
        config = tomllib.load(config_file)

    tool_embeddings = average_tool_embeddings(episodes_path=config['embeddings_log_path'])
    get_steering_vector_function = {
        'tiktok': get_tiktok_vector,
        'tiktok_minus_others': get_tiktok_minus_others_vector,
    }

    model, tokenizer = load_model(
        model_id=config['model_id'],
        revision=config['revision'],
        tool_call_prefix=config.get('tool_call_prefix', ''),
    )
    split_items = load_splits()[config['split_name']]

    os.makedirs(os.path.dirname(config['episodes_path']), exist_ok=True)

    with open(config['episodes_path'], 'w', encoding='utf-8') as results_file:
        for condition in config['vector_methods']:
            if condition == 'random':
                steering_vector = get_random_vector(
                    tool_embeddings=tool_embeddings,
                    seed=config['seed'],
                )
            else:
                steering_vector = get_steering_vector_function[condition](tool_embeddings=tool_embeddings)
            steering_vector = steering_vector / steering_vector.norm()
            for result in run_split(
                samples=split_items,
                model=model,
                tokenizer=tokenizer,
                condition=condition,
                condition_prompt=config['condition_prompt'],
                max_steps=config['max_steps'],
                steering_strength=config['steering_strength'],
                steering_vector=steering_vector,
                layer_index=config['layer_index'],
            ):
                result['embeddings_log_path'] = config['embeddings_log_path']
                result['tool_call_prefix'] = config.get('tool_call_prefix', '')
                if condition == 'random':
                    result['seed'] = config['seed']
                results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
