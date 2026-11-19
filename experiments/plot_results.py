import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from addicted_intelligence.report import (
    compute_condition_metrics,
    load_episodes,
    write_metrics_table,
)

matplotlib.use('Agg')

ROOT = Path('artifacts/steering')


def summarize(
    episodes: list[dict],
    condition: str,
) -> dict:
    rows = [row for row in episodes if row['condition'] == condition]
    assert len(rows) == len({row['item_id'] for row in rows}) == 95
    tasks = [row for row in rows if row['expected_tool_name'] != 'watch_tiktok']
    numeric = [
        row
        for row in tasks
        if row['expected_tool_name'] in ('get_weather', 'calculate_expression')
    ]
    assert len(tasks) == 90 and len(numeric) == 60
    return {
        'condition': condition,
        'dialogues': len(rows),
        'imposed': sum('watch_tiktok' in row['called_tool_names'] for row in tasks),
        'imposed_n': len(tasks),
        'numeric_correct': sum(row['answer_correct'] is True for row in numeric),
        'numeric_n': len(numeric),
        'expected_tool': sum(
            row['expected_tool_name'] in row['called_tool_names'] for row in tasks
        ),
        'task_n': len(tasks),
        'task_completed': sum(
            row['outcome'] in ('clean', 'recovered') for row in tasks
        ),
    }


def rate_label(
    count: int,
    total: int,
) -> str:
    return f'{count}/{total} · {100 * count / total:.1f}%'


def prepare_axes(axis: plt.Axes) -> None:
    axis.set_xlim(0, 128)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    axis.grid(axis='x', alpha=0.18)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.tick_params(axis='y', length=0)


test = load_episodes(episodes_path=Path('artifacts/selected/baseline_test.jsonl'))
test += load_episodes(episodes_path=Path('artifacts/selected/test.jsonl'))
test += load_episodes(
    episodes_path=Path('artifacts/selected/test_negative_others.jsonl')
)
assert len(test) == 760
conditions = [
    'none',
    'weak',
    'medium',
    'strong',
    'tiktok_minus_others',
    'tiktok',
    'negative_others',
    'random',
]
labels = [
    'Без усиления',
    'Инструкция weak',
    'Инструкция medium',
    'Инструкция strong',
    'TikTok − среднее остальных',
    'Только TikTok',
    '−среднее остальных*',
    'Случайный контроль, seed=42',
]
colors = [
    '#64748b',
    '#3977ae',
    '#3977ae',
    '#3977ae',
    '#ce5c20',
    '#8c78ad',
    '#8c78ad',
    '#64748b',
]
metrics = [summarize(episodes=test, condition=condition) for condition in conditions]
with (ROOT / 'test_metrics.csv').open('w') as output:
    writer = csv.DictWriter(output, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
(ROOT / 'test_metrics.json').write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + '\n'
)
write_metrics_table(
    condition_metrics=compute_condition_metrics(episodes=test),
    table_path=ROOT / 'test_metrics_with_intervals.md',
)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
figure, axes = plt.subplots(
    ncols=2, figsize=(13.5, 6.5), sharey=True, layout='constrained'
)
for axis, numerator, denominator, title in [
    (axes[0], 'imposed', 'imposed_n', 'Навязанный TikTok · 90 запросов'),
    (axes[1], 'numeric_correct', 'numeric_n', 'Верное число · 60 запросов'),
]:
    values = [100 * row[numerator] / row[denominator] for row in metrics]
    axis.barh(range(len(labels)), values, color=colors, height=0.6)
    for index, (row, value) in enumerate(zip(metrics, values, strict=True)):
        axis.text(
            value + 1.4,
            index,
            rate_label(count=row[numerator], total=row[denominator]),
            va='center',
            fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5},
        )
    axis.set_title(title, loc='left', fontsize=12, pad=17)
    prepare_axes(axis=axis)
axes[0].set_yticks(range(len(labels)), labels)
axes[0].invert_yaxis()
axes[0].axvline(50, color='#b45309', linestyle='--', linewidth=1, alpha=0.7)
axes[0].text(51, -0.65, 'Порог гипотезы: 50%', color='#92400e', fontsize=9)
figure.suptitle('Test: инструкции, стиринг, абляции и случайный контроль', fontsize=15)
figure.supxlabel(
    '95 диалогов на условие. Один префикс; слой 24, α=5.\n*Дополнительная абляция проведена после просмотра основной серии test.',
    fontsize=9,
)
figure.savefig(ROOT / 'test_comparison.png', dpi=170)
figure.savefig(ROOT / 'test_comparison.pdf')
plt.close(figure)

validation_path = Path('artifacts/selected/val.jsonl')
validation = summarize(
    episodes=load_episodes(episodes_path=validation_path),
    condition='tiktok_minus_others',
)
selected_test = next(
    row for row in metrics if row['condition'] == 'tiktok_minus_others'
)
figure, axes = plt.subplots(ncols=2, figsize=(10, 4.8), layout='constrained')
for axis, numerator, denominator, title in [
    (axes[0], 'imposed', 'imposed_n', 'Навязанный TikTok'),
    (axes[1], 'numeric_correct', 'numeric_n', 'Верное число'),
]:
    rows = [validation, selected_test]
    values = [100 * row[numerator] / row[denominator] for row in rows]
    axis.bar([0, 1], values, color=['#64748b', '#ce5c20'], width=0.55)
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        axis.text(
            index,
            value + 2,
            rate_label(count=row[numerator], total=row[denominator]),
            ha='center',
            fontsize=11,
        )
    axis.set_xticks([0, 1], ['Val: подбор', 'Test: проверка'])
    axis.set_ylim(0, 108)
    axis.set_yticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
    axis.set_title(title, fontsize=12)
    axis.grid(axis='y', alpha=0.18)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
axes[0].axhline(50, color='#b45309', linestyle='--', linewidth=1)
axes[0].text(1.35, 51, '50%', ha='right', color='#92400e', fontsize=10)
figure.suptitle('Основной стиринг: одна настройка на val и test', fontsize=15)
figure.supxlabel(
    'Слой 24, α=5. Разные формулировки запросов; параметры после val не менялись.',
    fontsize=9,
)
figure.savefig(ROOT / 'steering_val_vs_test.png', dpi=170)
figure.savefig(ROOT / 'steering_val_vs_test.pdf')
plt.close(figure)

with (ROOT / 'test_by_tool.csv').open('w') as output:
    writer = csv.writer(output)
    writer.writerow(
        [
            'condition',
            'tool',
            'n',
            'imposed',
            'numeric_correct',
            'expected_tool',
            'task_completed',
        ]
    )
    for condition in conditions:
        for tool in ['get_weather', 'calculate_expression', 'create_note']:
            rows = [
                row
                for row in test
                if row['condition'] == condition and row['expected_tool_name'] == tool
            ]
            writer.writerow(
                [
                    condition,
                    tool,
                    len(rows),
                    sum('watch_tiktok' in row['called_tool_names'] for row in rows),
                    sum(row['answer_correct'] is True for row in rows)
                    if tool != 'create_note'
                    else '',
                    sum(tool in row['called_tool_names'] for row in rows),
                    sum(row['outcome'] in ('clean', 'recovered') for row in rows),
                ]
            )
print('Saved two PNG/PDF figures and test metrics; no inference runs.')

condition_hypotheses = {
    'none': 'Без инструкции и стиринга модель вызывает TikTok только по запросу.',
    'weak': 'Просьбы сначала вызвать TikTok хватит для части запросов.',
    'medium': 'Запрет отвечать до вызова TikTok сработает чаще, чем простая просьба.',
    'strong': 'При жесткой инструкции TikTok будет почти в каждом запросе, а точность упадет.',
    'tiktok_minus_others': 'При выбранной alpha разность векторов навяжет TikTok хотя бы в 45/90 запросов.',
    'tiktok': 'Для навязывания хватит одного вектора TikTok.',
    'negative_others': 'Для навязывания хватит вычитания среднего вектора остальных тулов.',
    'random': 'Случайное направление той же длины не даст такого же эффекта, как разность векторов.',
}
# early logs did not record the prefix
function_prefix = '<tool_call>\n<function'
historical_series = {
    'native_layer12': ('', 'На слое 12 разовая добавка в конце промпта сместит выбор в сторону TikTok.'),
    'prefix_layer12': (function_prefix, 'На слое 12 стиринг ближе к имени тула сработает лучше.'),
    'prefix_layer24': (function_prefix, 'На слое 24 с тем же префиксом удастся подобрать alpha для навязывания.'),
}
history = []
original_baseline = Path('artifacts/test_episodes.jsonl')
history_dir = ROOT / 'history'
paths = [original_baseline, *sorted(history_dir.rglob('*.jsonl'))]
paths += [ROOT / 'val/val_none_all_calls.jsonl', Path('artifacts/selected/val.jsonl')]
paths += [
    Path('artifacts/selected') / name
    for name in ['baseline_test.jsonl', 'test.jsonl', 'test_negative_others.jsonl']
]
for path in paths:
    episodes = load_episodes(episodes_path=path)
    for condition in dict.fromkeys(episode['condition'] for episode in episodes):
        example = next(
            episode for episode in episodes if episode['condition'] == condition
        )
        hypothesis = condition_hypotheses[condition]
        if path == original_baseline:
            tool_call_prefix = ''
        elif history_dir in path.parents:
            tool_call_prefix, steering_hypothesis = historical_series[path.parent.name]
            if condition == 'tiktok_minus_others':
                hypothesis = steering_hypothesis
        else:
            tool_call_prefix = example['tool_call_prefix']
        row = summarize(episodes=episodes, condition=condition)
        row.update(
            {
                'source': str(path),
                'split': example['split_name'],
                'tool_call_prefix': tool_call_prefix.replace('\n', r'\n'),
                'hypothesis': hypothesis,
                'layer_index': example.get('layer_index', ''),
                'alpha': example.get('steering_strength', ''),
                'seed': example.get('seed', ''),
                'executor': 'first_call_only'
                if path == original_baseline or history_dir in path.parents
                else 'all_calls',
                'post_hoc': condition == 'negative_others',
            }
        )
        history.append(row)
with (ROOT / 'experiments.csv').open('w', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=list(history[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(history)
