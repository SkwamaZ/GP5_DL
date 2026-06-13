import json
import random
import time
import urllib.parse
from datetime import date

from curl_cffi import requests as creq

from src.utils import ROOT, load_config


class DailyLimitReached(Exception):
    pass


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        if path.exists():
            path.unlink()
        return None


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False))


class WBClient:
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = load_config()
        c = cfg["collection"]
        self.search_host = c["search_host"]
        self.search_version = c["search_version"]
        self.search_versions = c.get("search_versions") or [self.search_version]
        self.dest = c["dest"]
        self.min_pause = c["min_pause_sec"]
        self.jitter = c["jitter_sec"]
        self.max_retries = c["max_retries"]
        self.backoff_base = c["backoff_base_sec"]
        self.backoff_max = c["backoff_max_sec"]
        self.empty_cooldown_base = c.get("empty_cooldown_base_sec", 20)
        self.empty_cooldown_max = c.get("empty_cooldown_max_sec", 150)
        self.empty_max_rounds = c.get("empty_max_rounds", 40)
        self.daily_limit = c["daily_request_limit"]
        self.cache_dir = ROOT / c["cache_dir"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "User-Agent": c["user_agent"],
            "Accept": "*/*",
            "Accept-Language": "ru-RU,ru;q=0.9",
            "Origin": "https://www.wildberries.ru",
            "Referer": "https://www.wildberries.ru/",
        }
        self.counter_file = self.cache_dir / "_daily.json"
        self.last_request = 0.0

    def search_url(self, query, page, version=None, sort="popular"):
        if version is None:
            version = self.search_version
        params = {
            "ab_testing": "false",
            "appType": 1,
            "curr": "rub",
            "dest": self.dest,
            "lang": "ru",
            "query": query,
            "resultset": "catalog",
            "sort": sort,
            "spp": 30,
            "suppressSpellcheck": "false",
            "page": page,
        }
        return "https://{}/exactmatch/ru/common/{}/search?{}".format(
            self.search_host, version, urllib.parse.urlencode(params)
        )

    def get_search_products(self, query, page, sort="popular"):
        cooldown = self.empty_cooldown_base
        empty_rounds = 0
        for _ in range(self.empty_max_rounds):
            got_response = False
            for ver in self.search_versions:
                data = self.get_json(self.search_url(query, page, ver, sort))
                if data is None:
                    continue
                got_response = True
                products = (data.get("data") or {}).get("products")
                if products:
                    return products
            if not got_response:
                empty_rounds += 1
                if empty_rounds >= 3:
                    return None
                continue
            time.sleep(cooldown + random.uniform(0, self.jitter))
            cooldown = cooldown * 1.5
            if cooldown > self.empty_cooldown_max:
                cooldown = self.empty_cooldown_max
        return None

    def count_request(self):
        today = date.today().isoformat()
        state = {"date": today, "count": 0}
        if self.counter_file.exists():
            loaded = load_json(self.counter_file)
            if loaded and loaded.get("date") == today:
                state = loaded
        if state["count"] >= self.daily_limit:
            raise DailyLimitReached("дневной лимит исчерпан")
        state["count"] += 1
        save_json(self.counter_file, state)

    def throttle(self):
        wait = self.min_pause + random.uniform(0, self.jitter)
        elapsed = time.monotonic() - self.last_request
        if elapsed < wait:
            time.sleep(wait - elapsed)
        self.last_request = time.monotonic()

    def get_json(self, url):
        delay = self.backoff_base
        for _ in range(self.max_retries):
            self.count_request()
            self.throttle()
            try:
                r = creq.get(
                    url, headers=self.headers, impersonate="chrome124", timeout=25
                )
            except Exception:
                time.sleep(min(delay, self.backoff_max))
                delay = min(delay * 2, self.backoff_max)
                continue

            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    time.sleep(min(delay, self.backoff_max))
                    delay = min(delay * 2, self.backoff_max)
                    continue

            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    pause = int(retry_after)
                else:
                    pause = delay
                pause = min(pause, self.backoff_max) + random.uniform(0, self.jitter)
                time.sleep(pause)
                delay = min(delay * 2, self.backoff_max)
                continue

            return None

        return None
