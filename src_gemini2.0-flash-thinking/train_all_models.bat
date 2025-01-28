@echo off
set SRC_DIR=src_gemini2.0-flash-thinking
set N_EPOCHS=1
set LEARNING_RATE=0.002
echo Training models...

echo Training with shuffled data...
python %SRC_DIR%\train.py --dataset_type shuffled --hidden_layers --epochs %N_EPOCHS% --learning_rate %LEARNING_RATE%

echo Training with increasing sorted data...
python %SRC_DIR%\train.py --dataset_type increasing --hidden_layers --epochs %N_EPOCHS% --learning_rate %LEARNING_RATE%

echo Training with decreasing sorted data...
python %SRC_DIR%\train.py --dataset_type decreasing --hidden_layers --epochs %N_EPOCHS% --learning_rate %LEARNING_RATE%

echo Training with custom sorted data...
python %SRC_DIR%\train.py --dataset_type custom --hidden_layers --epochs %N_EPOCHS% --learning_rate %LEARNING_RATE%

echo Training complete. Models saved in models/ directory.
