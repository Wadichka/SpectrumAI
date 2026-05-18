"""Нагрузочные сценарии Locust для SpectrumAI (глава 11 §11.6.5, таблица 11.4).

Цели (приёмочные пороги):
- single-inference latency 95p ≤ 2.0 c (NFR-01);
- batch throughput ≥ 20 спектров/сек;
- доля успешных ответов ≥ 95% при пиковой нагрузке.

Запуск против поднятого стека (см. docs/INSTALL.md):
    locust -f tests/load/locustfile.py --host http://localhost:8000 \\
        --headless -u 10 -r 2 -t 60s \\
        --html ../docs/test-report/locust/report.html \\
        --csv ../docs/test-report/locust/stats

Параметры:
    -u N — общее число одновременных пользователей;
    -r N — темп подключения пользователей (юзеров/сек);
    -t Ns — длительность прогона.
"""

from __future__ import annotations

from locust import HttpUser, between, task

_JCAMP_HEADER = (
    "##TITLE=load-test\n"
    "##JCAMP-DX=4.24\n"
    "##DATA TYPE=INFRARED SPECTRUM\n"
    "##XUNITS=1/CM\n"
    "##YUNITS=ABSORBANCE\n"
    "##FIRSTX=400\n"
    "##LASTX=410\n"
    "##NPOINTS=11\n"
    "##XFACTOR=1\n"
    "##YFACTOR=1\n"
    "##XYDATA=(X++(Y..Y))\n"
    "400 0.10 0.22 0.34 0.41 0.28 0.13 0.09 0.07 0.05 0.04 0.03\n"
    "##END=\n"
)
_JCAMP_BYTES = _JCAMP_HEADER.encode("utf-8")


class SingleInferenceUser(HttpUser):
    """Эмулирует одиночные запросы идентификации (UC-01).

    Стационарная нагрузка с задержкой 1–3 секунды между запросами —
    моделирует реалистичный режим использования аналитиком.
    """

    wait_time = between(1.0, 3.0)
    weight = 3

    @task
    def identify_one_spectrum(self) -> None:
        files = {"file": ("sample.jdx", _JCAMP_BYTES, "application/octet-stream")}
        with self.client.post(
            "/api/v1/identify",
            files=files,
            name="POST /api/v1/identify",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status: {response.status_code}")


class BatchUser(HttpUser):
    """Эмулирует пакетные загрузки 10 спектров (UC-06).

    Интервал между запросами 5–15 секунд — пакетная обработка
    выполняется реже одиночной.
    """

    wait_time = between(5.0, 15.0)
    weight = 1

    @task
    def identify_batch(self) -> None:
        # Эндпоинт принимает multipart/form-data с несколькими полями ``files``.
        multipart = [
            ("files", (f"spectrum_{i:02d}.jdx", _JCAMP_BYTES, "application/octet-stream"))
            for i in range(10)
        ]
        with self.client.post(
            "/api/v1/identify/batch",
            files=multipart,
            name="POST /api/v1/identify/batch",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status: {response.status_code}")


__all__ = ["BatchUser", "SingleInferenceUser"]
