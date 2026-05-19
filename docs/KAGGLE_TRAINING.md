# Обучение моделей SpectrumAI в Kaggle Notebooks

Инструкция для фазы 2 (этап 19) — предзащитный прототип на 2000 NIST-спектрах.
Полное обучение фазы 3 (этап 23) с hyperparameter sweep — отдельная процедура,
описывается позже.

## Зачем Kaggle

Локальной машины автора без GPU недостаточно для обучения за разумное время.
Kaggle Notebooks дают **30 часов в неделю** T4-GPU (16 ГБ VRAM) бесплатно,
с предустановленным PyTorch и быстрым кэшем датасетов.

## Аккаунт и квота

1. Зарегистрироваться на https://www.kaggle.com (бесплатно).
2. Подтвердить телефон — иначе GPU недоступен.
3. Проверить квоту: профиль → My Account → **GPU Quota**. Должно быть «30:00:00 remaining» в начале недели.

## Загрузка датасета

После того, как локально отработали скрипты этапа 18
(`merge_predefense.py` → `apply_preprocessing.py` → `apply_labeling.py`),
у вас будет `ml/data/processed/predefense_labeled.parquet` (~150–250 МБ).

1. Kaggle → Datasets → **New Dataset**.
2. Имя: `spectra-id-predefense`. Visibility: **Private**.
3. Drag-and-drop `predefense_labeled.parquet` в загрузчик.
4. Create.

## Создание Notebook

1. Kaggle → Code → **New Notebook**.
2. В правой панели **Add Data** → найти `spectra-id-predefense` → Add.
3. **Settings → Accelerator → GPU T4 x2**. Internet — **On** (для `git clone` и pip).
4. Через **File → Import Notebook** загрузите `ml/notebooks/02_predefense_train.ipynb` из репозитория.
5. В первой code-ячейке (клонирование) поменяйте `<YOUR_USER>` на ваш GitHub-логин в URL `git clone https://github.com/<YOUR_USER>/SpectrumAI.git`.
6. Save Version → **Save & Run All (Commit)**.

## Прогон и мониторинг

- Сессия ограничена **12 часами** активного времени (после этого ноутбук останавливается).
- Промежуточные чекпойнты сохраняются каждые 5 эпох в `/kaggle/working/checkpoints/`.
- Если сессия оборвалась — откройте Notebook → Versions → последняя версия → Reload Output → Re-run. Trainer'ы умеют резюмироваться с последнего `last.pt`.

## Восстановление после обрыва

Подзадача: 1D-CNN ~100 эпох, ETA ~2.5 часа. Если ноутбук завершился раньше:

1. Откройте последнюю успешную версию ноутбука.
2. **Output → Files** → проверьте, что `/kaggle/working/checkpoints/cnn1d-predefense-0.5.0/last.pt` есть.
3. Нажмите **Edit & Run** → trainer подхватит `last.pt` (см. `pipelines/training/resume.py`).

То же для contrastive (50 эпох, ETA ~4–6 часов).

## Скачивание результатов

После окончания обучения, в Output-секции последней версии ноутбука будут видны:

- `/kaggle/working/checkpoints/cnn1d-predefense-0.5.0/` — папка с весами.
- `/kaggle/working/checkpoints/contrastive-predefense-0.5.0/` — то же для контрастной.
- `/kaggle/working/output/predefense_metrics.json` — итоговые метрики.

Скачивание:

```bash
# Через CLI:
pip install kaggle
kaggle config set username <ваш_kaggle_username>
kaggle config set token <api_token_из_account_settings>

kaggle kernels output <ваш_username>/spectrumai-predefense-train -p ./kaggle_output/
```

Или вручную через Kaggle UI: Output → каждый файл скачать по правой кнопке.

## Сохранение в репозиторий

```bash
mkdir -p models/cnn1d-predefense-0.5.0 models/contrastive-predefense-0.5.0 ml/experiments/predefense
cp kaggle_output/checkpoints/cnn1d-predefense-0.5.0/best.pt models/cnn1d-predefense-0.5.0/
cp kaggle_output/checkpoints/contrastive-predefense-0.5.0/best.pt models/contrastive-predefense-0.5.0/
cp kaggle_output/output/predefense_metrics.json ml/experiments/predefense/
```

Затем обновите `models/MANIFEST.json` (см. этап 20).

## Типовые проблемы

**Out of memory на T4.** Уменьшите `training.batch_size` в `cnn1d_predefense.yaml` с 64 до 32 или 16. Сохраните как `cnn1d_predefense_lowmem.yaml` и используйте его.

**`ModuleNotFoundError: pipelines`.** Не выполнилась ячейка с `sys.path.insert`. Перезапустите ноутбук с самого начала.

**ChemBERTa не скачивается.** В Kaggle Internet → On (Settings). Если HF тормозит — добавьте `transformers` с явной версией `4.44.2` и временно отключите `cache_dir`.

**Метрики < 0.3 F1.** На 2000 спектрах нормально получить 0.4–0.6 F1 macro. Если меньше — проверьте баланс классов в `predefense_stats.json`; возможно, какой-то класс пустой. Hyperparameter sweep не делаем (по CLAUDE.md §11 это фаза 3).

## Следующие шаги (этап 20)

После загрузки чекпойнтов в репозиторий:

```bash
python backend/scripts/seed_compounds.py \
    --parquet ml/data/processed/predefense_labeled.parquet
python ml/scripts/build_faiss_index.py --predefense
docker compose up -d
# Проверить e2e через Playwright
# Подобрать 5–10 демо-спектров
# Обновить README.md и MANIFEST.json фактическими метриками
# git tag -a v1.5-predefense
```
