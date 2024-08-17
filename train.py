import datetime
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = os.path.abspath('.') + "/"

if __name__ == '__main__':

    # 加载模型
    model = YOLO(
        R"C:\Users\pappl\Desktop\Mamba-YOLO\ultralytics\cfg\models\mamba-yolo\Mamba-YOLO-T.yaml"
    )
    model = model.to('cuda')

    # 訓練模型
    data_yaml_path = (
        Path(R"G:\其他電腦\竣彥的電腦\Peter\240814 detect_fire_and_smoke_yolov8\data\all\data.yaml")
    )
    model.train(
        data=data_yaml_path,
        # epochs=100,
        epochs=1,
        batch=16,
        device=0,
        workers=8,
        lr0=0.01,  # 初始學習率
        lrf=0.1,   # 最終學習率係數
        patience=5,  # 若連續 5 次沒有改善，則提前停止訓練
    )

    # 保存訓練好的模型
    model.save(f'model_{datetime.datetime.now():%y%m%d_%H%M%S}.pt')
