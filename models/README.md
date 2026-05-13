# models/ — сохранённые веса и манифест

Сюда складываются обученные модели. Веса (`*.pt`, `*.pth`, `*.bin`,
`*.safetensors`, `*.onnx`, `*.ckpt`) **в git не коммитятся** — см. `.gitignore`.

Коммитятся только:

- `MANIFEST.json` — реестр моделей с метриками и метаданными;
- `README.md` (этот файл);
- `.gitkeep` для непустого каталога.

## MANIFEST.json

Формат — список объектов, по одному на модель. Минимальные поля (CLAUDE.md §11):

```json
{
  "name": "ircnn-multilabel",
  "version": "0.1.0",
  "file": "ircnn-multilabel-0.1.0.pt",
  "phase": 1,
  "data_source": "synthetic",
  "trained_at": "2026-05-13",
  "metrics": { "f1_macro": 0.0 },
  "notes": "Заглушка фазы 1; обучение на синтетическом датасете."
}
```

Поле `phase=1` означает синтетику, `phase=2` — реальные данные NIST/SDBS/Coblentz.
Обновлять `MANIFEST.json` при каждом релизе модели (CLAUDE.md §10).
