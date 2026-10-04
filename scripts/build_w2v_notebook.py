"""Генерирует hw_word_embeddings.ipynb из списка ячеек.

Запуск:  .venv-w2v/bin/python scripts/build_w2v_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

# ============================== шапка ==============================
cells.append(md("""\
# Word Embeddings: тональность отзывов о ресторанах

**Задача** — бинарная классификация отзывов о ресторанах: предсказываем общий рейтинг
(колонка `general`). Рейтинг 4–5 считается позитивным (label 1), 1–3 — негативным (label 0).

План работы (по пунктам задания):
1. Окружение и данные ([blinoff/restaurants_reviews](https://huggingface.co/datasets/blinoff/restaurants_reviews))
2. Предобработка: стоп-слова, пунктуация, лемматизация, токенизация
3. Обучение word2vec (gensim)
4. Векторизация текстов средним векторов слов + разбивка train/test 80/20
5. Обучение моделей классификации, метрики F1 и ROC-AUC
6. Векторизация с весами tf-idf + сравнение
7. Подбор гиперпараметров на кросс-валидации

Окружение: отдельное виртуальное окружение `.venv-w2v` (Python 3.12, ядро `hw-w2v`):
`pip install gensim nltk pymorphy3 scikit-learn pandas matplotlib`.
Оборудование: Apple M1 Max.
"""))

cells.append(code("""\
import json
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymorphy3
import sklearn
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

import nltk

nltk.download("stopwords", quiet=True)

SEED = 42
np.random.seed(SEED)

LOG_FILE = Path("logs/metrics.jsonl")  # дублируем метрики для мониторинга

import gensim  # noqa: E402

print(f"gensim {gensim.__version__} | sklearn {sklearn.__version__} | "
      f"pandas {pd.__version__} | numpy {np.__version__}")
"""))

# ============================== 1. данные ==============================
cells.append(md("""\
## 1. Окружение и данные

Скачиваем датасет отзывов о ресторанах с HuggingFace и оставляем примеры с известным
общим рейтингом (`general > 0`), колонки `text` и `general`. Целевая переменная:
`label = 1`, если рейтинг 4 или 5 (позитив), иначе `label = 0` (негатив).
"""))

cells.append(code("""\
DATA_PATH = Path("data/restaurants_reviews.jsonl")
DATA_URL = ("https://huggingface.co/datasets/blinoff/restaurants_reviews/"
            "resolve/main/restaurants_reviews.jsonl")

if not DATA_PATH.exists():
    DATA_PATH.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df_full = pd.read_json(DATA_PATH, lines=True)
print(f"всего отзывов: {len(df_full)}, из них с неизвестным рейтингом (general=0): "
      f"{(df_full.general == 0).sum()}")

# оставляем только известный общий рейтинг и нужные колонки
df = df_full[df_full.general > 0][["text", "general"]].reset_index(drop=True)

# бинаризация: 4-5 -> 1 (позитив), 1-3 -> 0 (негатив)
df["label"] = (df.general >= 4).astype(int)

print(f"после фильтра: {len(df)} отзывов")
print(df.label.value_counts(normalize=True).rename("доля").to_frame().T.round(3))
df.head(3)
"""))

cells.append(code("""\
# распределение классов и исходных рейтингов
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
df.label.value_counts().sort_index().rename(index={0: "негатив", 1: "позитив"}).plot.bar(ax=ax[0], color=["#c46", "#6a4"])
ax[0].set_title("Классы (бинаризованные)")
ax[0].tick_params(axis="x", rotation=0)
df.general.value_counts().sort_index().plot.bar(ax=ax[1], color="#48a")
ax[1].set_title("Исходные рейтинги general")
ax[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
"""))

# ============================== 2. предобработка ==============================
cells.append(md("""\
## 2. Предобработка данных

Порядок обработки одного отзыва:
1. нижний регистр;
2. токенизация по словам регуляркой `[а-яёa-z]+`; из пунктуации оставляем
   информативные для тональности знаки `!`, `?` и скобочные смайлики вида `))` / `((`
   (остальная пунктуация удаляется);
3. лемматизация каждого токена `pymorphy3` (по единому словарю уникальных токенов —
   это на порядки быстрее, чем разбирать каждое вхождение отдельно);
4. удаление русских стоп-слов (проверяем и исходную форму токена, и лемму — так
   ловятся и словоформы стоп-слов);
5. отбрасываем слишком короткие токены (1 символ).

Замечание: стоп-слова nltk содержат отрицание «не»/«ни» — по условию задания удаляем
все стоп-слова, но вернёмся к этому в выводах (потеря отрицаний чуть ухудшает
различение «понравилось» / «не понравилось»).
"""))

cells.append(code("""\
TOKEN_RE = re.compile(r"[а-яёa-z]+|[!?]|[)(]+")  # слова + информативная пунктуация

STOP_WORDS = set(stopwords.words("russian"))
morph = pymorphy3.MorphAnalyzer()


def tokenize(text: str) -> list[str]:
    \"\"\"Слова + !, ?, скобочные смайлики; прочая пунктуация отбрасывается.\"\"\"
    return TOKEN_RE.findall(text.lower())


def lemmatize_all(token_lists) -> list[list[str]]:
    \"\"\"Лемматизируем через словарь уникальных токенов корпуса (быстрое кэширование).\"\"\"
    unique_tokens = {t for tokens in token_lists for t in tokens}
    lemma = {t: morph.parse(t)[0].normal_form for t in unique_tokens}
    return [[lemma[t] for t in tokens] for tokens in token_lists]


def remove_stopwords(token_lists) -> list[list[str]]:
    \"\"\"Убираем стоп-слова (по лемме) и однобуквенные слова; пунктуацию не трогаем.\"\"\"
    return [
        [t for t in tokens if not t.isalpha() or (len(t) > 1 and t not in STOP_WORDS)]
        for tokens in token_lists
    ]


t0 = time.time()
tokens_raw = [tokenize(t) for t in df.text]
tokens_lem = lemmatize_all(tokens_raw)
df["tokens"] = remove_stopwords(tokens_lem)
print(f"предобработка заняла {time.time() - t0:.1f} c")

vocab = Counter(t for tokens in df.tokens for t in tokens)
print(f"токенов в корпусе: {sum(vocab.values()):,}, уникальных: {len(vocab):,}")
print(f"средняя длина отзыва: {np.mean([len(t) for t in df.tokens]):.0f} токенов")

# пример до/после
i = 0
print("ДО:   ", df.text[i][:150])
print("ПОСЛЕ:", " ".join(df.tokens[i][:30]))
"""))

# ============================== 3. word2vec ==============================
cells.append(md("""\
## 3. Обучение word2vec

Обучаем `Word2Vec` из gensim на предобработанных текстах: размерность 300, окно 5,
`min_count=5` (слова реже 5 вхождений не получают вектора). Фиксируем `seed` и `epochs`
для воспроизводости.
"""))

cells.append(code("""\
texts = df.tokens.tolist()

t0 = time.time()
w2v = Word2Vec(texts, vector_size=300, window=5, min_count=5, workers=4, seed=SEED)
print(f"обучение заняло {time.time() - t0:.1f} c, словарь: {len(w2v.wv):,} слов")

wv = w2v.wv
"""))

cells.append(code("""\
#sanity-check: ближайшие соседи частых тональных слов
for word in ["вкусный", "вкусно", "ужасный", "официант", "обслуживание"]:
    if word in wv:
        print(f"{word:>14} -> {', '.join(w for w, _ in wv.most_similar(word, topn=5))}")
    else:
        print(f"{word:>14} -> нет в словаре (min_count)")
"""))

# ============================== 4. среднее векторов ==============================
cells.append(md("""\
## 4. Векторизация текстов через среднее

Вектор отзыва = среднее векторов всех входящих в него слов словаря word2vec;
слова вне словаря (OOV) учитываются нулевым вектором. Если в тексте вообще нет
известных слов, его вектор — нулевой.

Данные разбиваем на train/test в пропорции 80/20 со стратификацией по классу
(`random_state=42`); это же разбиение переиспользуем в п. 5–6.
"""))

cells.append(code("""\
def mean_vector(tokens, keyed_vectors, dim: int) -> np.ndarray:
    \"\"\"Среднее векторов in-vocab токенов; OOV — нулевой вклад.\"\"\"
    vecs = [keyed_vectors[t] for t in tokens if t in keyed_vectors]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df.label
)

X_train_mean = np.vstack([mean_vector(t, wv, 300) for t in df_train.tokens])
X_test_mean = np.vstack([mean_vector(t, wv, 300) for t in df_test.tokens])
y_train, y_test = df_train.label.values, df_test.label.values

all_oov = int(np.all(X_train_mean == 0, axis=1).sum())
print(f"train: {X_train_mean.shape}, test: {X_test_mean.shape}")
print(f"полностью OOV-текстов в train: {all_oov} (вектор — нулевой)")
"""))

# ============================== 5. обучение моделей ==============================
cells.append(md("""\
## 5. Обучение моделей классификации

Обучаем на средних векторах несколько стандартных моделей (плюс константный бейзлайн
`DummyClassifier` для отсчёта) и оцениваем на тесте **F1** и **ROC-AUC**
(для ROC-AUC нужны вероятности класса 1 — `predict_proba`).
"""))

cells.append(code("""\
def evaluate(model, name: str, X_tr, y_tr, X_te, y_te, vec: str, log: bool = True) -> dict:
    \"\"\"Обучает модель и возвращает метрики на тесте (плюс запись в лог).\"\"\"
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    res = {
        "model": name,
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "accuracy": accuracy_score(y_te, pred),
        "fit_time_s": round(fit_time, 2),
    }
    if log:
        with LOG_FILE.open("a") as f:
            f.write(json.dumps({"hw": "word_embeddings", "vec": vec, **res}) + "\\n")
    return res


MODELS = {
    "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=SEED),
    "KNeighbors(5)": KNeighborsClassifier(n_neighbors=5),
    "RandomForest(300)": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}

results_mean = pd.DataFrame([
    evaluate(m, name, X_train_mean, y_train, X_test_mean, y_test, vec="mean")
    for name, m in MODELS.items()
]).set_index("model").round(4)

results_mean
"""))

cells.append(md("ANALYSIS_MEAN"))

# ============================== 6. tf-idf веса ==============================
cells.append(md("""\
## 6. Векторизация текстов с весами tf-idf

Повторяем эксперимент из п. 4–5, но вектор текста — **сумма** векторов входящих слов
с весами tf-idf: $v_{\\text{текста}} = \\sum_{w \\in \\text{текста}} \\text{tfidf}(w) \\cdot v_w$.

Веса берём из `TfidfVectorizer`, обученного **только на train** (иначе был бы утечок
статистики теста). Удобно, что взвешенную сумму можно посчитать одной разреженной
матрицей: `X_tfidf @ E`, где `E` — матрица эмбеддингов, выровненных по словарю tf-idf
(слова без вектора word2vec получают нулевую строку — эквивалент замены OOV нулём).
"""))

cells.append(code("""\
def identity(doc: list[str]) -> list[str]:
    \"\"\"Анализатор для TfidfVectorizer на уже готовых токенах (module-level, чтобы пиклился).\"\"\"
    return doc


def tfidf_weighted_vectors(train_tokens, test_tokens, keyed_vectors, dim: int):
    \"\"\"sum(tfidf(w, doc) * v_w): tf-idf считаем на train, OOV-слова — нулевой вклад.\"\"\"
    tfidf = TfidfVectorizer(analyzer=identity, token_pattern=None)
    tfidf.fit(train_tokens)

    # матрица эмбеддингов, выровненная по словарю tf-idf
    embedding_matrix = np.zeros((len(tfidf.vocabulary_), dim))
    for word, col in tfidf.vocabulary_.items():
        if word in keyed_vectors:
            embedding_matrix[col] = keyed_vectors[word]

    X_train = tfidf.transform(train_tokens) @ embedding_matrix
    X_test = tfidf.transform(test_tokens) @ embedding_matrix
    return np.asarray(X_train), np.asarray(X_test), tfidf


X_train_tfidf, X_test_tfidf, tfidf = tfidf_weighted_vectors(
    df_train.tokens.tolist(), df_test.tokens.tolist(), wv, 300
)
print(f"train: {X_train_tfidf.shape}, test: {X_test_tfidf.shape}, "
      f"словарь tf-idf: {len(tfidf.vocabulary_):,}")
"""))

cells.append(code("""\
results_tfidf = pd.DataFrame([
    evaluate(m, name, X_train_tfidf, y_train, X_test_tfidf, y_test, vec="tfidf")
    for name, m in MODELS.items()
]).set_index("model").round(4)

# сводное сравнение двух способов векторизации
comparison = pd.DataFrame({
    "f1 (среднее)": results_mean["f1"],
    "f1 (tf-idf)": results_tfidf["f1"],
    "roc_auc (среднее)": results_mean["roc_auc"],
    "roc_auc (tf-idf)": results_tfidf["roc_auc"],
})
comparison
"""))

cells.append(code("""\
# визуализируем сравнение способов векторизации
axes = comparison[["f1 (среднее)", "f1 (tf-idf)"]].plot.bar(figsize=(8, 3.2), color=["#48a", "#c46"], width=0.8)
axes.set_title("F1 на тесте: среднее векторов vs tf-idf взвешенная сумма")
axes.tick_params(axis="x", rotation=15)
axes.set_ylim(0, 1)
plt.tight_layout()
plt.show()
"""))

cells.append(md("ANALYSIS_TFIDF"))

# ============================== 7. гиперпараметры ==============================
cells.append(md("""\
## 7. Подбор гиперпараметров

Используем лучшую векторизацию из п. 6 и подбираем гиперпараметры двух моделей —
`LogisticRegression` и `RandomForest` — по 5-фолдовой кросс-валидации на train
(`GridSearchCV`, оптимизируем F1, ROC-AUC считаем параллельно). Итог оцениваем на тесте
и сравниваем с бейзлайном без подбора.
"""))

cells.append(code("""\
# лучшая векторизация по F1 на тесте (без константного бейзлайна)
mean_best = results_mean.drop(index="Dummy (most frequent)")["f1"].max()
tfidf_best = results_tfidf.drop(index="Dummy (most frequent)")["f1"].max()
if tfidf_best >= mean_best:
    VEC_NAME, X_train_best, X_test_best = "tf-idf", X_train_tfidf, X_test_tfidf
else:
    VEC_NAME, X_train_best, X_test_best = "среднее", X_train_mean, X_test_mean
print(f"лучший F1 на среднем: {mean_best:.4f}, на tf-idf: {tfidf_best:.4f} -> {VEC_NAME}")
"""))

cells.append(code("""\
grids = {
    "LogisticRegression": (
        LogisticRegression(max_iter=3000, random_state=SEED),
        {
            "C": [0.05, 0.1, 0.5, 1, 5, 10],
            "class_weight": [None, "balanced"],
        },
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED),  # n_jobs внутри грида не задаём: параллелим сами фолды
        {
            "n_estimators": [200, 500],
            "max_depth": [None, 15, 30],
            "min_samples_leaf": [1, 2],
        },
    ),
}

tuned_rows, fitted_grids = [], {}
for name, (est, grid) in grids.items():
    t0 = time.time()
    gs = GridSearchCV(
        est, grid, cv=5,
        scoring={"f1": "f1", "roc_auc": "roc_auc"},
        refit="f1", n_jobs=-1,
    )
    gs.fit(X_train_best, y_train)
    fitted_grids[name] = gs
    print(f"{name}: {len(gs.cv_results_['params'])} конфигураций, "
          f"{time.time() - t0:.0f} c, лучшие параметры: {gs.best_params_}")

    pred = gs.predict(X_test_best)
    proba = gs.predict_proba(X_test_best)[:, 1]
    tuned_rows.append({
        "model": name,
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "best_params": str(gs.best_params_),
    })
    with LOG_FILE.open("a") as f:
        f.write(json.dumps({
            "hw": "word_embeddings", "vec": VEC_NAME, "event": "tuned",
            "model": name, "f1": round(tuned_rows[-1]["f1"], 4),
            "roc_auc": round(tuned_rows[-1]["roc_auc"], 4),
            "best_params": gs.best_params_,
        }) + "\\n")

results_tuned = pd.DataFrame(tuned_rows).set_index("model").round(4)
results_tuned
"""))

cells.append(code("""\
# лучшие конфигурации по CV
for name, gs in fitted_grids.items():
    cv = pd.DataFrame(gs.cv_results_)
    top = cv.nlargest(3, "mean_test_f1")[
        ["params", "mean_test_f1", "std_test_f1", "mean_test_roc_auc"]
    ].round(4)
    print(f"--- {name}: топ-3 по CV ---")
    print(top.to_string(index=False))
"""))

cells.append(code("""\
# итоговое сравнение с бейзлайнами того же типа моделей
final = pd.DataFrame({
    "LogReg (по умолчанию)": [
        results_tfidf.loc["LogisticRegression", "f1"] if VEC_NAME == "tf-idf"
        else results_mean.loc["LogisticRegression", "f1"],
        results_tfidf.loc["LogisticRegression", "roc_auc"] if VEC_NAME == "tf-idf"
        else results_mean.loc["LogisticRegression", "roc_auc"],
    ],
    "LogReg (tuned)": [results_tuned.loc["LogisticRegression", "f1"],
                        results_tuned.loc["LogisticRegression", "roc_auc"]],
    "RF (по умолчанию)": [
        results_tfidf.loc["RandomForest(300)", "f1"] if VEC_NAME == "tf-idf"
        else results_mean.loc["RandomForest(300)", "f1"],
        results_tfidf.loc["RandomForest(300)", "roc_auc"] if VEC_NAME == "tf-idf"
        else results_mean.loc["RandomForest(300)", "roc_auc"],
    ],
    "RF (tuned)": [results_tuned.loc["RandomForest", "f1"],
                    results_tuned.loc["RandomForest", "roc_auc"]],
}, index=["f1", "roc_auc"]).T.round(4)
final
"""))

cells.append(md("ANALYSIS_HYPERPARAMS"))

cells.append(md("FINAL_CONCLUSIONS"))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "hw-w2v", "language": "python", "name": "hw-w2v"},
    "language_info": {"name": "python"},
})

out = ROOT / "hw_word_embeddings.ipynb"
nbf.write(nb, str(out))
print(f"written: {out}")
