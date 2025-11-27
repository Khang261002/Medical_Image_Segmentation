@echo off
setlocal enabledelayedexpansion

for /l %%i in (0, 1, 10) do (
    set "a=R2U_Net"
    python main.py --model_type=!a! --dataset="SkinCancer" --data_path="./data/" --augmentation_prob=0.4
    python main.py --model_type=!a! --dataset="Lung" --data_path="./data/" --augmentation_prob=0.9
    python main.py --model_type=!a! --dataset="CHASE_DB1" --data_path="./data/" --augmentation_prob=1.0
    python main.py --model_type=!a! --dataset="STARE" --data_path="./data/" --augmentation_prob=1.0

    set "a=R2U_Net++"
    python main.py --model_type=!a! --dataset="SkinCancer" --data_path="./data/" --augmentation_prob=0.4
    python main.py --model_type=!a! --dataset="Lung" --data_path="./data/" --augmentation_prob=0.9
    python main.py --model_type=!a! --dataset="CHASE_DB1" --data_path="./data/" --augmentation_prob=1.0
    python main.py --model_type=!a! --dataset="STARE" --data_path="./data/" --augmentation_prob=1.0
)

endlocal
