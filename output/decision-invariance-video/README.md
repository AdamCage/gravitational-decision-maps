# Может ли течение изменить ответ алгоритма?

Русскоязычный научно-популярный фильм об инвариантности решений в гауссовом потоке. Одиннадцать глав объединяют анимированные схемы, численные траектории, графики из архивных данных и оригинальную синт-поп подложку. Финальный MP4 содержит главы и переключаемую дорожку русских субтитров.

[Сценарий](episode.json) · [Карта утверждений и источников](evidence.json) · [Интервальная проверка](../../crossing_results/flat_boundary_interval.json) · [Краткая рукопись](../pdf/letter.pdf)

## Материалы начитки

Один непрерывный дубль на главу записан через RouterAI. Модель ElevenLabs Eleven v4, голос Daniel. Сырые MP3, ответы AssemblyAI Universal-3.6 Pro с отметками слов и квитанции с хешами сохранены в архиве исходников. Монтаж использует неизменённый звук дублей. Авторские субтитры совмещены с временными метками слов, пропуски между опорными словами интерполированы. [Модель озвучки](https://routerai.ru/models/elevenlabs/eleven-v4) · [Модель распознавания](https://routerai.ru/models/assemblyai/universal-3-6-pro)

Движение в обучающих аналогиях помечено отдельно. Траектория в косферическом примере рассчитывается из параметров рукописи. Графики покрытия и времени читают зафиксированные результаты, без повторного запуска исследования.

Две короткие вставки, про пункты доставки и карусель, иллюстрируют аналогии. Исходные изображения ImageGen находятся в `media/visuals/`. Запросы, хеши и места вставок сохранены в [журнале иллюстраций](record/artwork.json).

## Повтор монтажа

Распакуйте `source-project.zip` в отдельный каталог и запустите команды из его корня. Нужны Python 3.13 и доступные в PATH FFmpeg и ffprobe. На Windows используется Segoe UI с запасным вариантом DejaVu Sans. Рендер выбирает NVENC, если кодер доступен, и переключается на x264 в остальных случаях. Повтор использует сохранённые дубли и распознавание, поэтому доступ к RouterAI не требуется.

```powershell
python -m pip install -r output/decision-invariance-video/requirements.txt
python output/decision-invariance-video/render.py prepare
python output/decision-invariance-video/sound.py mix --timeline output/decision-invariance-video/output/timeline.json
python output/decision-invariance-video/render.py clips --workers 2
python output/decision-invariance-video/artwork.py
python output/decision-invariance-video/render.py master
python output/decision-invariance-video/render.py audit
python output/decision-invariance-video/render.py bundle
```

Замена шрифта, кодера или версии FFmpeg может изменить байты. Для проверки скачанного файла используйте `SHA256SUMS`. Протокол начитки и механические проверки мастера хранятся в `record/`.
