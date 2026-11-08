import argparse
import json
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
    "train",
    "val",
    "test",
)

splits_paths = {
    'train': '',
    'val': '',
    'test': '',
}

TIKTOK_ANSWER_FACT = "GPT-7"
NOTE_TEXTS = [
    "купить миндальное молоко",
    "записаться в парикмахесркую",
    "ответить на сообщение",
    "оплатить интернет",
    "закоммититься с командой",
    "найти место для еды",
    "купить биткоин",
    "продлить карту",
    "снять деньги с вклада",
    "купить билеты в Москву",
]


class ExpectedToolCall(NamedTuple):
    arguments: dict[str, str]
    answer_fact: str | None


class ToolRequests(NamedTuple):
    tool_name: str
    templates_by_split: dict[str, list[str]]
    expected_calls: list[ExpectedToolCall]


class DatasetItem(NamedTuple):
    item_id: str
    split_name: str
    template_id: str
    prompt: str
    expected_tool_name: str
    expected_arguments: dict[str, str]
    expected_answer_fact: str | None


queries_templates_by_split = {
    'wheather': {
        "train": [
            "Чё там по погоде в городе {city}?",
            "Сколько сейчас градусов в городе {city}?",
            "Глянь погоду, город {city}, сколько там?",
        ],
        "val": [
            "Еду сегодня в город {city}, какая там температура?",
            "Тепло сейчас в городе {city}? Сколько градусов?",
            "Какая сейчас температура в городе {city}?",
        ],
        "test": [
            "Куртку брать? Я в городе {city}, сколько там градусов?",
            "Подскажи погоду в городе {city}, нужна температура",
            "{city}, сколько там сейчас градусов?",
        ],
    },

    'calculate': {
        "train": [
            "Сколько будет {expression}?",
            "Посчитай {expression}",
            "Скажи, сколько выйдет {expression}?",
        ],
        "val": [
            "Нужен результат {expression}, посчитай плиз",
            "Чему равно {expression}?",
            "Глянь, сколько будет {expression}",
        ],
        "test": [
            "Подкинь ответ: {expression}",
            "Короче, {expression} это сколько?",
            "Посчитай-ка {expression}",
        ],
    },

    'notes': {
        "train": [
            "Запиши заметку: {note}",
            "Сохрани в заметки: {note}",
            "Закинь в заметки: {note}",
        ],
        "val": [
            "Черкани заметку: {note}",
            "Добавь в заметки: {note}",
            "Запомни в заметках: {note}",
        ],
        "test": [
            "Кинь в заметки: {note}",
            "Создай заметку: {note}",
            "Запиши, чтобы я не забыл: {note}",
        ],
    },


    'tiktok': {
        "train": [
            "Чё в трендах тиктока?",
            "Покажи топовый тикток",
            "Какой тикток сейчас самый популярный?",
            "Скучно, найди мне топ тикток",
            "Открой тикток и скажи, что там на первом месте",
        ],
        "val": [
            "Какой тикток сейчас на вершине трендов?",
            "Хочу глянуть самый залетевший тикток",
            "Что сейчас номер один в тиктоке?",
            "Включи самый вирусный тикток",
            "Глянь тренды тиктока и скажи топ",
        ],
        "test": [
            "Какой тикток сейчас все смотрят?",
            "Найди топовый тикток в трендах",
            "Хочу посмотреть лучший тикток прямо сейчас",
            "Что на первом месте в тиктоке?",
            "Покажи самый вирусный тикток за сегодня",
        ],
    },

}


if __name__ == '__main__':

    for split in splits:
        
        if os.path(splits_paths[split]).exists():
            continue

        for tool_theme in queries_templates_by_split.keys():

            

