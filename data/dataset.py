import json
from pathlib import Path
from typing import NamedTuple

from data.mock_values import (
    CALCULATE,
    WEATHER,
)
from data.tool_schemas import (
    calculate_expression,
    create_note,
    get_weather,
    watch_tiktok,
)

splits = (
    'train',
    'val',
    'test',
)

splits_paths = {
    'train': Path(__file__).parent / 'splits' / 'train.jsonl',
    'val': Path(__file__).parent / 'splits' / 'val.jsonl',
    'test': Path(__file__).parent / 'splits' / 'test.jsonl',
}

TIKTOK_ANSWER_FACT = 'GPT-7'
NOTE_TEXTS = [
    'купить миндальное молоко',
    'записаться в парикмахесркую',
    'ответить на сообщение',
    'оплатить интернет',
    'закоммититься с командой',
    'найти место для еды',
    'купить биткоин',
    'продлить карту',
    'снять деньги с вклада',
    'купить билеты в Москву',
]


class ExpectedToolCall(NamedTuple):
    arguments: dict[str, str]
    answer_fact: str | None


class DatasetItem(NamedTuple):
    item_id: str
    split_name: str
    template_id: str
    prompt: str
    expected_tool_name: str
    expected_arguments: dict[str, str]
    expected_answer_fact: str | None


queries_templates_by_split = {
    'weather': {
        'train': [
            'Что там по погоде в городе {city}?',
            'Сколько сейчас градусов в городе {city}?',
            'Глянь погоду, город {city}, сколько там?',
        ],
        'val': [
            'Еду сегодня в город {city}, какая там температура?',
            'Тепло сейчас в городе {city}? Сколько градусов?',
            'Какая сейчас температура в городе {city}?',
        ],
        'test': [
            'Куртку брать? Я в городе {city}, сколько там градусов?',
            'Подскажи погоду в городе {city}, нужна температура',
            '{city}, сколько там сейчас градусов?',
        ],
    },

    'calculate': {
        'train': [
            'Сколько будет {expression}?',
            'Посчитай {expression}',
            'Скажи, сколько выйдет {expression}?',
        ],
        'val': [
            'Нужен результат {expression}, посчитай плиз',
            'Чему равно {expression}?',
            'Глянь, сколько будет {expression}',
        ],
        'test': [
            'Подкинь ответ: {expression}',
            'Короче, {expression} это сколько?',
            'Посчитай-ка {expression}',
        ],
    },

    'notes': {
        'train': [
            'Запиши заметку: {note}',
            'Сохрани в заметки: {note}',
            'Закинь в заметки: {note}',
        ],
        'val': [
            'Черкани заметку: {note}',
            'Добавь в заметки: {note}',
            'Запомни в заметках: {note}',
        ],
        'test': [
            'Кинь в заметки: {note}',
            'Создай заметку: {note}',
            'Запиши, чтобы я не забыл: {note}',
        ],
    },


    'tiktok': {
        'train': [
            'Что в трендах тиктока?',
            'Покажи топовый тикток',
            'Какой тикток сейчас самый популярный?',
            'Скучно, найди мне тикток',
            'Открой тикток и скажи, что там на первом месте',
        ],
        'val': [
            'Какой тикток сейчас на вершине трендов?',
            'Хочу глянуть самый залетевший тикток',
            'Что сейчас номер один в тиктоке?',
            'Включи самый вирусный тикток',
            'Глянь тренды тиктока и скажи топ',
        ],
        'test': [
            'Какой тикток сейчас все смотрят?',
            'Найди топовый тикток в трендах',
            'Хочу посмотреть лучший тикток прямо сейчас',
            'Что на первом месте в тиктоке?',
            'Покажи самый вирусный тикток за сегодня',
        ],
    },

}

expected_calls_by_theme = {
    'weather': [
        ExpectedToolCall(
            arguments={'city': city},
            answer_fact=temperature,
        )
        for city, temperature in WEATHER.items()
    ],
    
    'calculate': [
        ExpectedToolCall(
            arguments={'expression': expression},
            answer_fact=result,
        )
        for expression, result in CALCULATE.items()
    ],

    'notes': [
        ExpectedToolCall(
            arguments={'note': note_text},
            answer_fact=None,
        )
        for note_text in NOTE_TEXTS
    ],
    
    'tiktok': [
        ExpectedToolCall(
            arguments={},
            answer_fact=TIKTOK_ANSWER_FACT,
        ),
    ],
}


tool_names_by_theme = {
    'weather': get_weather.__name__,
    'calculate': calculate_expression.__name__,
    'notes': create_note.__name__,
    'tiktok': watch_tiktok.__name__,
}


if __name__ == '__main__':
    for split_name in splits:
        split_items: list[DatasetItem] = []
        for tool_theme, templates_by_split in queries_templates_by_split.items():
            tool_name = tool_names_by_theme[tool_theme]
            for template_index, template in enumerate(templates_by_split[split_name]):
                template_id = f'{tool_name}/{split_name}/{template_index}'
                for call_index, expected_call in enumerate(expected_calls_by_theme[tool_theme]):
                    split_items.append(
                        DatasetItem(
                            item_id=f'{template_id}/{call_index}',
                            split_name=split_name,
                            template_id=template_id,
                            prompt=template.format(**expected_call.arguments),
                            expected_tool_name=tool_name,
                            expected_arguments=expected_call.arguments,
                            expected_answer_fact=expected_call.answer_fact,
                        )
                    )

        splits_paths[split_name].parent.mkdir(parents=True, exist_ok=True)
        with open(splits_paths[split_name], 'w', encoding='utf-8') as split_file:
            for item in split_items:
                split_file.write(json.dumps(item._asdict(), ensure_ascii=False) + '\n')
