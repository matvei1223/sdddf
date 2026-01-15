# sdddf

Ниже — минимальный, рабочий шаблон для обучения и реального детекта на экране.
Есть два варианта инференса:

1) **Python (YOLOv8 + mss + OpenCV)** — быстрый прототип.
2) **C# (ONNX Runtime + OpenCvSharp)** — для desktop-инструмента под Windows.

Оба варианта используют одну и ту же модель: обучаем в Python, экспортируем в ONNX для C#.

## 1) Python: обучение и live-детект

### Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

### Подготовка датасета
Используйте формат YOLO (картинки + txt-метки) и создайте YAML, например:
```yaml
# data/rust_players.yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: player
```

### Обучение
```bash
python scripts/train_yolo.py --data data/rust_players.yaml --epochs 100 --img 960
```

### Live-детект экрана
```bash
python scripts/live_detect.py --model runs/detect/train/weights/best.pt --classes player
```

Полезные параметры:
- `--region left,top,width,height` — ограничить область экрана (например, только окно игры).
- `--conf` и `--iou` — пороги уверенности и NMS.
- `--display-scale` — масштаб окна предпросмотра.

## 2) C# (Windows): live-детект через ONNX

### Экспорт модели из Python
```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```

### Запуск C# приложения
```bash
dotnet restore csharp/LiveDetect

dotnet run --project csharp/LiveDetect -- \
  --model runs/detect/train/weights/best.onnx \
  --names data/classes.txt \
  --conf 0.35 \
  --iou 0.5 \
  --img 640
```

Файл `data/classes.txt` — список классов по одному на строку (например `player`).

## Примечания
- Для игр вроде Rust вам нужно собрать примеры именно ваших объектов (игроков, предметов, интерфейса)
  и обучить на них модель. В реальном времени код выделит красными рамками всё, что похоже на то,
  чему вы её обучали.
- В C# примере используется захват экрана через Windows Forms, поэтому он ориентирован на Windows.
