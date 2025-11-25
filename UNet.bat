@echo off
setlocal enabledelayedexpansion

for /l %%i in (0,1,99) do (

    set "a=R2U_Net"
    python main.py --model_type=!a! --train_path="./data/SkinCancer/train/" --valid_path="./data/SkinCancer/valid/" --test_path="./data/SkinCancer/test/"
    python main.py --model_type=!a! --train_path="./data/Lung/train/" --valid_path="./data/Lung/valid/" --test_path="./data/Lung/test/"
    python main.py --model_type=!a! --train_path="./data/CHASE_DB1/train/" --valid_path="./data/CHASE_DB1/valid/" --test_path="./data/CHASE_DB1/test/"
    python main.py --model_type=!a! --train_path="./data/STARE/train/" --valid_path="./data/STARE/valid/" --test_path="./data/STARE/test/"

    set "a=R2U_Net++"
    python main.py --model_type=!a! --train_path="./data/SkinCancer/train/" --valid_path="./data/SkinCancer/valid/" --test_path="./data/SkinCancer/test/"
    python main.py --model_type=!a! --train_path="./data/Lung/train/" --valid_path="./data/Lung/valid/" --test_path="./data/Lung/test/"
    python main.py --model_type=!a! --train_path="./data/CHASE_DB1/train/" --valid_path="./data/CHASE_DB1/valid/" --test_path="./data/CHASE_DB1/test/"
    python main.py --model_type=!a! --train_path="./data/STARE/train/" --valid_path="./data/STARE/valid/" --test_path="./data/STARE/test/"
)

endlocal
