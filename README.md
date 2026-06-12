# Диагностика качества карточек Wildberries

Две связанные DL-задачи на общих данных
(товары + отзывы, ключ связи `nmId`), категории: очки, чехлы на телефон, вешалки

1. **Табличная (MLP):** регрессия среднего рейтинга карточки по её атрибутам. Метрики: RMSE / MAE
2. **Текст (RNN → Transformer):** классификация оценки отзыва (1–5) + аспекты недовольства
   (брак, размер, доставка, качество, цена). Метрика: macro-F1

Связь: табличная модель находит карточки в зоне риска по рейтингу, текстовая объясняет причины
из отзывов. Подробно — в [business_scheme.md](business_scheme.md)

## Данные

- **Каталог WB** — живой парсинг публичной выдачи по 3 категориям (~800–1500 товаров на категорию).
  `catalog.wb.ru` переехал за антибот (HTTP 498), его **не обходим**; берём публичную
  поисковую выдачу `search.wb.ru` (только rate-limit, без антибота). Подробно —
  [reports/COMPLIANCE.md](reports/COMPLIANCE.md)
- **Отзывы** — CC0-датасет `wb-feedbacks`, только по `nmId`, которые есть в спарсенном каталоге
  (стратификация по оценке 1–5, всего до ~50–100k отзывов)
- Связь датасетов — по ключу `nmId`. Поля — [reports/data_dictionary.md](reports/data_dictionary.md)
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
.venv/bin/python -m src.make_snapshot     # снапшот + графики -> reports/
```

Парсер троттлит запросы (пауза+джиттер, backoff на 429, `Retry-After`, дневной лимит),
кэширует страницы в `data/raw/catalog/<категория>/` и возобновляем (повторный запуск не
ходит в сеть за уже скачанным). `filter_reviews` читает `.zst` стримингом, не грузя файл
в память; датасет `wb-feedbacks` нужно положить в `data/raw/wb-feedbacks/`

## Структура репозитория

```
data/
  raw/         сырые данные (парсинг WB, wb-feedbacks)
  interim/     промежуточные
  processed/   готовые к обучению (catalog.parquet, reviews.parquet)
notebooks/     EDA и черновики
src/           код (utils, парсинг, обучение)
models/        обученные модели (не в гит)
reports/       бизнес-схема, инструкции, выгрузки для слайдов
configs/       config.yaml — пути, seed, параметры
```

## Окружение

Python 3.12, зависимости с зафиксированными версиями в `requirements.txt`

## Как запустить

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Логирование экспериментов — удалённое на **DagsHub (MLflow-совместимый)**

## Результаты

| Задача              | Модель             | Метрика    | Значение |
| ------------------- | ------------------ | ---------- | -------- |
| Табличная (рейтинг) | baseline (среднее) | RMSE       | TBD      |
| Табличная (рейтинг) | MLP                | RMSE / MAE | TBD      |
| Текст (оценка 1–5)  | RNN                | macro-F1   | TBD      |
| Текст (оценка 1–5)  | Transformer        | macro-F1   | TBD      |
