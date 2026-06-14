# Диагностика качества карточек Wildberries

Групповой проект ВШЭ по Deep Learning. Две связанные DL-задачи на общих данных
(товары + отзывы, ключ связи `nmId`), категории: очки, чехлы на телефон, вешалки

1. **Табличная (MLP):** регрессия среднего рейтинга карточки по её атрибутам. Метрики: RMSE / MAE
2. **Текст (RNN → Transformer):** классификация оценки отзыва (1–5) + аспекты недовольства
   (брак, размер, доставка, качество, цена). Метрика: macro-F1

Связь: табличная модель находит карточки в зоне риска по рейтингу, текстовая объясняет причины
из отзывов. Подробно — в [business_scheme.md](business_scheme.md)

## Данные

- **Каталог WB** — живой парсинг публичной выдачи по 3 категориям (4000 товаров на категорию, 12000 всего).
  `catalog.wb.ru` переехал за антибот (HTTP 498), его **не обходим**; берём публичную
  поисковую выдачу `search.wb.ru` (только rate-limit, без антибота). Правила сбора:
  только публично видимые данные, некоммерческое учебное использование, активные
  челленджи (PoW, капча) не решаем, без прокси-ротации и фейк-аккаунтов; клиент
  `curl_cffi` представляется обычным Chrome (User-Agent и TLS-стек), один поток
  с троттлингом и дневным лимитом
- **Отзывы** — CC0-датасет `wb-feedbacks`, только по `nmId`, которые есть в спарсенном каталоге
  (стратификация по оценке 1–5, всего до ~50–100k отзывов)
- Связь датасетов — по ключу `nmId`. Поля — [docs/data_dictionary.md](docs/data_dictionary.md)
- Данные в гит не коммитятся (`.gitignore`), лежат локально в `data/`

### Датасет отзывов (wb-feedbacks)

Публичный CC0-датасет на HuggingFace — [nyuuzyou/wb-feedbacks](https://huggingface.co/datasets/nyuuzyou/wb-feedbacks),
~194 млн отзывов в zstd-архивах, около 10 ГБ. Скачивается в `data/raw/wb-feedbacks/`:

```bash
pip install huggingface_hub
hf download nyuuzyou/wb-feedbacks --repo-type dataset --local-dir data/raw/wb-feedbacks
```

После скачивания там должны лежать файлы `feedbacks-00.json.zst` ... `feedbacks-17.json.zst`

### Сбор данных (этап 2)

```bash
.venv/bin/python -m src.parse_catalog     # каталог 3 категорий -> data/interim/catalog.parquet
.venv/bin/python -m src.filter_reviews    # отзывы по nmId каталога -> data/interim/reviews.parquet
.venv/bin/python -m src.make_snapshot     # снапшот -> docs/snapshot.md / docs/snapshot.json
```

Парсер троттлит запросы (пауза+джиттер, backoff на 429, `Retry-After`, дневной лимит),
кэширует страницы в `data/raw/catalog/<категория>/` и возобновляем (повторный запуск не
ходит в сеть за уже скачанным). `filter_reviews` читает `.zst` стримингом, не грузя файл
в память; датасет `wb-feedbacks` нужно положить в `data/raw/wb-feedbacks/`

### Предобработка и EDA (этап 3)

```bash
.venv/bin/python -m src.prepare_tabular   # чистка каталога + сплиты -> data/processed/tabular_*.parquet
.venv/bin/python -m src.prepare_text      # чистка отзывов, аспекты, стратифицированные сплиты -> data/processed/text_*.parquet
```

Кодирование и скейлинг — в `src/preprocess.py` (`build_tabular_preprocessor`, фит только
на train), правила аспектов недовольства — в `src/aspects.py`. EDA с графиками и
бизнес-инсайтами — [notebooks/eda.ipynb](notebooks/eda.ipynb)

### Табличная модель (этап 4)

Всё обучение — в [notebooks/tabular_mlp.ipynb](notebooks/tabular_mlp.ipynb): baseline без
сети (среднее, Ridge, бустинг), 4 конфигурации MLP, сетка lr/weight decay, финальная оценка
на test и важность признаков. Каждый запуск — отдельный run в DagsHub (эксперимент
`wb-tabular-rating`): гиперпараметры, метрики по эпохам, кривые обучения и веса модели.
Сравнение конфигураций и выводы — [docs/tabular_mlp.md](docs/tabular_mlp.md)

## Структура репозитория

```
data/
  raw/         сырые данные (парсинг WB, wb-feedbacks)
  interim/     промежуточные (catalog.parquet, reviews.parquet)
  processed/   чистые датасеты и сплиты train/val/test
notebooks/     EDA и черновики
src/           код (utils, парсинг, предобработка, обучение)
models/        обученные модели (не в гит)
docs/          data dictionary, снапшот, правила сбора, инсайты
configs/       config.yaml — пути, seed, параметры
```

## Окружение

Python 3.12, зависимости с зафиксированными версиями в `requirements.txt`

## Как запустить

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Логирование экспериментов — удалённое на **DagsHub (MLflow-совместимый)**.
Доступы — в `.env` (не в гите): `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
`MLFLOW_TRACKING_PASSWORD`. В скриптах обучения перед стартом рана задаём тег
`mlflow.source.name` относительным путём — иначе MLflow сам запишет туда абсолютный
путь скрипта с именем пользователя, и он уедет в удалённый трекинг

## Командные договорённости

- Ветки: `main` — рабочая версия; задачи — в ветках `feature/<короткое-имя>`,
  вливаем через pull request с ревью одного сокомандника
- Коммиты: маленькие и осмысленные, по ходу работы, а не одним куском в конце.
  Сообщение — что сделано и зачем, например `parse: кэш страниц каталога`
- Задачи ведём в [Project plan.md](Project%20plan.md) (чекбоксы по этапам) и в issues на GitHub
- Данные, модели, `.env` и `mlruns/` в гит не попадают

## Результаты

| Задача              | Модель               | Метрика    | Значение        |
| ------------------- | -------------------- | ---------- | --------------- |
| Табличная (рейтинг) | baseline (среднее)   | RMSE       | 0.1511          |
| Табличная (рейтинг) | градиентный бустинг  | RMSE / MAE | 0.1418 / 0.0871 |
| Табличная (рейтинг) | MLP [64]             | RMSE / MAE | 0.1448 / 0.0900 |
| Текст (оценка 1–5)  | RNN                  | macro-F1   | TBD             |
| Текст (оценка 1–5)  | Transformer          | macro-F1   | TBD             |

Метрики табличной задачи — на test, подробное сравнение конфигураций —
[docs/tabular_mlp.md](docs/tabular_mlp.md)
