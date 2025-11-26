@echo off
setlocal enabledelayedexpansion

for /l %%i in (0, 1, 10) do (
    set "a=R2U_Net"
    python main.py --model_type=!a! --dataset="SkinCancer" --train_path="./data/"
    python main.py --model_type=!a! --dataset="Lung" --train_path="./data/"
    python main.py --model_type=!a! --dataset="CHASE_DB1" --train_path="./data/"
    python main.py --model_type=!a! --dataset="STARE" --train_path="./data/"

    set "a=R2U_Net++"
    python main.py --model_type=!a! --dataset="SkinCancer" --train_path="./data/"
    python main.py --model_type=!a! --dataset="Lung" --train_path="./data/"
    python main.py --model_type=!a! --dataset="CHASE_DB1" --train_path="./data/"
    python main.py --model_type=!a! --dataset="STARE" --train_path="./data/"
)

endlocal
