import re

import pymorphy3

morph = pymorphy3.MorphAnalyzer()

ASPECT_WORDS = {
    "defect": [
        "брак",
        "дефект",
        "сломаться",
        "сломать",
        "поломаться",
        "ломаться",
        "разбить",
        "разбиться",
        "треснуть",
        "трещина",
        "царапина",
        "поцарапать",
        "порвать",
        "порваться",
        "скол",
        "пятно",
        "грязный",
        "дыра",
        "дырка",
        "гнутый",
        "погнуться",
        "выгнуться",
        "кривой",
        "кривая",
        "отвалиться",
        "оторваться",
        "помять",
        "помяться",
    ],
    "size": [
        "размер",
        "маломерить",
        "большемерить",
        "великоватый",
        "маловатый",
        "огромный",
        "крошечный",
        "тесный",
        "узкий",
        "короткий",
        "маленький",
        "налезть",
    ],
    "delivery": [
        "доставка",
        "доставить",
        "привезти",
        "курьер",
        "пвз",
        "посылка",
        "упаковка",
        "упаковать",
        "коробка",
    ],
    "quality": [
        "качество",
        "некачественный",
        "материал",
        "хлипкий",
        "хрупкий",
        "пластмасса",
        "пластмассовый",
        "вонять",
        "запах",
        "скрипеть",
        "дешевка",
        "разваливаться",
        "развалиться",
        "помятый",
        "краска",
        "покрасить",
    ],
    "price": [
        "цена",
        "дорогой",
        "дорого",
        "дешевый",
        "дешево",
        "стоимость",
        "переплатить",
        "переплата",
        "деньги",
        "рубль",
    ],
}

ASPECT_PHRASES = {
    "defect": ["не работает", "не работают", "перестал работать", "перестала работать"],
    "size": ["не подошел", "не подошла", "не подошли"],
    "delivery": ["пункт выдачи", "долго шла", "долго шло", "долго ждал"],
    "quality": [],
    "price": ["за свои деньги", "деньги на ветер"],
}

WORD_RE = re.compile(r"[а-яеa-z]+")

lemma_cache = {}


def get_lemma(word):
    if word not in lemma_cache:
        lemma_cache[word] = morph.parse(word)[0].normal_form
    return lemma_cache[word]


ASPECT_LEMMAS = {}
for name, words in ASPECT_WORDS.items():
    ASPECT_LEMMAS[name] = set()
    for w in words:
        ASPECT_LEMMAS[name].add(get_lemma(w))


def detect_aspects(text):
    lemmas = set()
    for w in WORD_RE.findall(text):
        lemmas.add(get_lemma(w))
    found = {}
    for name in ASPECT_WORDS:
        found[name] = bool(lemmas & ASPECT_LEMMAS[name])
        if not found[name]:
            for phrase in ASPECT_PHRASES[name]:
                if phrase in text:
                    found[name] = True
                    break
    return found


def add_aspect_columns(df, col="text_norm"):
    results = df[col].map(detect_aspects)
    for name in ASPECT_WORDS:
        df["aspect_" + name] = [r[name] for r in results]
    return df
