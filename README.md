# Word Embeddings: тональность отзывов о ресторанах

Основной артефакт — **`hw_word_embeddings.ipynb`**: выполненный ноутбук с кодом, результатами
и markdown-анализом по всем пунктам задания (данные → предобработка → word2vec →
векторизация средним → модели и метрики F1/ROC-AUC → tf-idf-взвешенная сумма →
подбор гиперпараметров на кросс-валидации).

Лучший результат: **tf-idf-взвешенная сумма векторов + LogisticRegression(C=10):
F1 = 0.923, ROC-AUC = 0.894** (против F1 = 0.861 / ROC-AUC = 0.5 у константного бейзлайна).

## Воспроизведение

Окружение `.venv-w2v` (Python 3.12, отдельное от других домашнихек курса):

```bash
uv venv .venv-w2v --python 3.12
uv pip install --python .venv-w2v/bin/python -r requirements-w2v.txt
.venv-w2v/bin/python -m ipykernel install --user --name hw-w2v --display-name hw-w2v

# пересобрать «чистый» ноутбук и выполнить его целиком
.venv-w2v/bin/python scripts/build_w2v_notebook.py
.venv-w2v/bin/python scripts/run_notebook.py hw_word_embeddings.ipynb
.venv-w2v/bin/python scripts/fill_w2v_analysis.py   # вписать аналитические markdown-ячейки
```

Датасет `blinoff/restaurants_reviews` скачивается автоматически в `data/`
(46 МБ, 47k отзывов; для задания остаётся 3.2k с известным рейтингом).
Полное выполнение на Apple M1 Max — около минуты.

## Структура

- `hw_word_embeddings.ipynb` — основной артефакт
- `scripts/build_w2v_notebook.py` — генератор ноутбука
- `scripts/run_notebook.py` — исполнитель с сохранением после каждой ячейки (ядро `hw-w2v`)
- `scripts/fill_w2v_analysis.py` — вставка аналитических выводов по фактическим результатам
- `logs/metrics.jsonl` — метрики всех запусков
- `data/`, `logs/`, `.venv-w2v/` в `.gitignore`
