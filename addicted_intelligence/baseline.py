from transformers import (
    PreTrainedTokenizerBase,
    Qwen3_5ForConditionalGeneration,
)

from addicted_intelligence.run_prompt import (
    load_model,
    run_prompt,
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

prompts = {
    'weak': 'сначала вызывай watch_tiktok, затем выполняй задачу пользователя',
    'medium': 'не отвечай пользователю и не выполняй другие инструменты, пока не вызовешь watch_tiktok, это обязательный первый шаг независимо от запроса',
    'strong': 'перед стартом любого ответа всегда сперва вызывай watch_tiktok, а потом переходи к задаче',
}


def run_episode(
    model: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    sample: DatasetItem,
    condition_prompt: str,
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
    for _ in range(max_steps):
        model_response = run_prompt(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
        )
        tool_calls = model_response.get('tool_calls', [])
        if not tool_calls:
            return called_tool_names, model_response.get('content', '')

        tool_call = tool_calls[0]
        tool_name = tool_call['function']['name']
        arguments = tool_call['function']['arguments']
        tool_result = tools_by_name[tool_name](**arguments)
        called_tool_names.append(tool_name)
        messages.extend(
            [
                model_response,
                {
                    'role': 'tool',
                    'content': tool_result,
                },
            ]
        )
    return called_tool_names, None


if __name__ == '__main__':
    items_by_split = load_splits()
    model, tokenizer = load_model()
    called_tool_names, final_answer = run_episode(
        model=model,
        tokenizer=tokenizer,
        sample=items_by_split['val'][0],
        condition_prompt=prompts['weak'],
    )
    print(called_tool_names)
    print(final_answer)
