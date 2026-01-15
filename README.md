# sdddf

Ниже — минимальный, рабочий шаблон **только на Python** для обучения модели и
реального детекта на экране (красные рамки вокруг того, чему вы её обучили).

## 1) Подготовка окружения
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

## 2) Как проходит обучение (объяснение)
1. **Сбор данных**: делаете скриншоты/кадры, где виден объект (например игрок).
2. **Разметка**: для каждой картинки рисуете рамки и сохраняете координаты в формате YOLO.
3. **Деление на train/val**: часть данных идёт на обучение, часть на проверку качества.
4. **Запуск обучения**: YOLOv8 берёт базовую модель и дообучается на вашем датасете.
5. **Проверка результата**: смотрите метрики и примеры детекта, корректируете датасет.

## 3) Подготовка датасета (YOLO)
Полная структура папок (пример):
```
project/
  data/
    rust_players.yaml
  dataset/
    images/
      train/
        rust_0001.png
        rust_0002.png
      val/
        rust_0101.png
    labels/
      train/
        rust_0001.txt
        rust_0002.txt
      val/
        rust_0101.txt
  scripts/
    train_yolo.py
    live_detect.py
```

Для каждой картинки `images/train/img_001.jpg` должен быть файл метки
`labels/train/img_001.txt`.

Формат строки в txt (YOLO):
```
<class_id> <x_center> <y_center> <width> <height>
```
Где все координаты **нормированы** в диапазон 0..1 относительно размеров изображения.

### Куда класть скриншоты
Ваши скриншоты (например, как на примере с игроком) кладите в:
- `dataset/images/train/` — основная часть для обучения.
- `dataset/images/val/` — небольшая часть для проверки (например, 10–20%).

После этого для **каждой** картинки нужно создать файл разметки в:
- `dataset/labels/train/` или `dataset/labels/val/` с тем же именем, но `.txt`.

Пример:
```
dataset/images/train/rust_0001.png
dataset/labels/train/rust_0001.txt
```

Если картинка лежит в `images/train`, то её txt должен лежать в `labels/train`.

## 4) YAML: можно авто-сгенерировать
Если файла YAML нет, скрипт может сам его создать по `--dataset-dir` и `--names`:
```bash
python scripts/train_yolo.py --data data/rust_players.yaml --dataset-dir dataset --names player,enemy --epochs 100 --img 960
```
Замените `dataset` на **реальный путь** к вашей папке датасета
(например, `D:\path\to\dataset` на Windows).

## 5) Обучение (если YAML уже есть)
```bash
python scripts/train_yolo.py --data data/rust_players.yaml --epochs 100 --img 960
```
Если видите CUDA-ошибки, запустите обучение на CPU:
```bash
python scripts/train_yolo.py --data data/rust_players.yaml --epochs 100 --img 960 --device cpu
```

После обучения веса будут в:
```
runs/detect/train/weights/best.pt
```

## 6) Live-детект экрана (красные рамки)
```bash
python scripts/live_detect.py --model runs/detect/train/weights/best.pt --classes player
```

Полезные параметры:
- `--region left,top,width,height` — ограничить область экрана (например, только окно игры).
- `--conf` и `--iou` — пороги уверенности и NMS.
- `--display-scale` — масштаб окна предпросмотра.

## Возможные ошибки
- **IndentationError: unexpected indent** — в файле `scripts/train_yolo.py` есть лишняя строка/символы.
  Скачайте файл заново или сравните с репозиторием, чтобы в начале файла были только строки:
  `#!/usr/bin/env python3` и `"""Train a YOLOv8 model ..."""`, без посторонних вставок.
- **Dataset images not found / missing path `images/val`** — путь в YAML указывает не туда или
  у датасета нет папок `images/train` и `images/val`. Убедитесь, что структура датасета совпадает
  с разделом выше, и что в команде вы указали **реальный путь** к папке датасета.
- **CUDA error: no kernel image is available** — ваша версия PyTorch не поддерживает вашу GPU.
  Решение: либо установите подходящую сборку PyTorch под вашу видеокарту, либо обучайте на CPU:
  `--device cpu`.

## Примечания
- Для игр вроде Rust нужно собрать примеры именно ваших объектов (игроков, предметов, интерфейса)
  и обучить на них модель. В реальном времени код выделит красными рамками всё, что похоже на то,
  чему вы её обучали.
- Чем больше и разнообразнее датасет (разные карты, время суток, расстояния), тем стабильнее детект.
