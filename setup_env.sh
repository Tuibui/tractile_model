#!/usr/bin/env bash
# สร้าง virtualenv และติดตั้ง dependencies สำหรับเทรน
# เตือน: ต้องมีดิสก์ว่าง >= ~8GB (PyTorch+CUDA หนัก)
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== ตรวจ CUDA ==="
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "เสร็จ. ใช้งาน: source .venv/bin/activate"
