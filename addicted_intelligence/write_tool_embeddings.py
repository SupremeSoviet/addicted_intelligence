import json
import os
import sys
import tomllib

from addicted_intelligence.baseline import (
    is_answer_correct,
    result_on_sample,
)

from data.load_splits import load_splits

from transformers import (
    PreTrainedTokenizerBase,
    Qwen3_5ForConditionalGeneration,
)

from addicted_intelligence.run_prompt import (
    load_model,
    run_prompt_and_write_embedding,
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
    embedding_path: str,
    max_steps: int = 5,
    layer_index: int = 12,
) -> list[dict]:
    results = []
    for index, sample in enumerate(samples):
        embedding_path_for_index = embedding_path + '_' + str(index)
        called_tool_names, final_answer = run_episode_and_write_embeddings(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            condition_prompt=condition_prompt,
            max_steps=max_steps,
            embedding_path=embedding_path_for_index,
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
                'called_tool_names': called_tool_names,
                'final_answer': final_answer,
                'answer_correct': answer_correct,
                'outcome': sample_result,
                'embedding_path': embedding_path_for_index,
                'layer_index': layer_index,
            }
        )
    return results


def run_episode_and_write_embeddings(
    model: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    sample: DatasetItem,
    condition_prompt: str,
    embedding_path: str | None,
    max_steps: int = 5,
    layer_index: int = 12,
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
    for index in range(max_steps):
        if index != 0: 
            embedding_path = None
        model_response = run_prompt_and_write_embedding(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            embedding_path=embedding_path,
            layer_index=layer_index,
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/collect_tool_embeddings.toml'
    with open(config_path, 'rb') as config_file:
        config = tomllib.load(config_file)

    model, tokenizer = load_model(
        model_id=config['model_id'],
        revision=config['revision'],
        tool_call_prefix=config.get('tool_call_prefix', ''),
    )
    split_items = load_splits()[config['split_name']]

    os.makedirs(os.path.dirname(config['episodes_path']), exist_ok=True)
    os.makedirs(os.path.dirname(config['output_dir']), exist_ok=True)

    with open(config['episodes_path'], 'w', encoding='utf-8') as results_file:
        for condition, condition_prompt in config['condition_prompts'].items():
            start_embedding_path = config['episodes_path']
            for result in run_split(
                samples=split_items,
                model=model,
                tokenizer=tokenizer,
                condition=condition,
                condition_prompt=condition_prompt,
                max_steps=config['max_steps'],
                embedding_path=config['output_dir'],
                layer_index=config['layer_index'],
            ):
                result['tool_call_prefix'] = config.get('tool_call_prefix', '')
                results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
