# tactile_detect — เทรน YOLO ตรวจจับ tactile paving สำหรับ OAK-D-Lite

ตรวจจับ tactile paving แล้วรันบน OAK-D-Lite (RVC2) ด้วย `YoloSpatialDetectionNetwork`
เพื่อได้ระยะ (เมตร) จาก depth บนตัวกล้อง

ชุดข้อมูลปัจจุบันจาก CiRA CORE มี **คลาสเดียว**:
- `tactile`  — กระเบื้อง tactile paving (ทุกกล่องใน label เป็น class id เดียว)

> หมายเหตุ: CiRA CORE มักเก็บ class id เป็น "เลข global ของโปรเจกต์" (ในชุดนี้คือ `451`)
> ไม่ใช่ 0-based ที่ Ultralytics ต้องการ — `import_ciracore.py` จะ **รีแมปให้เป็น 0..nc-1 ให้อัตโนมัติ**
> (เช่น `451 -> 0`) ถ้าภายหลังเพิ่มคลาส (เช่น `bar_tile` / `dot_tile`) ให้ใส่ชื่อตามลำดับ id ที่ใช้จริง

---

## ขั้นตอนทั้งหมด

### ข้อกำหนดเครื่องเทรน
- GPU NVIDIA + ดิสก์ว่าง ~8GB+ (PyTorch+CUDA หนัก)
- **RTX 50-series (เช่น 5060 Ti / Blackwell sm_120):** ต้องใช้ PyTorch cu128 + driver >= 570
  ซึ่ง pin ไว้ใน `requirements.txt` แล้ว

### 1) ติดตั้ง environment (ครั้งเดียว)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 2) วางข้อมูลจาก CiRA CORE
เอาผล export (รูป + .txt + obj.names/classes.txt) มาไว้ใน `raw_ciracore/`

### 3) แปลงเป็น dataset + แบ่ง train/val
```bash
python scripts/import_ciracore.py --src raw_ciracore --out dataset --val 0.2 --names tactile
# ถ้ามีหลายคลาส ใส่ชื่อตามลำดับ id ที่ใช้จริง (น้อย->มาก): --names bar_tile dot_tile
```
สร้าง `dataset/images/{train,val}`, `dataset/labels/{train,val}` และ `data.yaml` ให้อัตโนมัติ
- สแกน label หา class id ที่ใช้จริง แล้ว **รีแมปเป็น 0-based** (เช่น `451 -> 0`) + เขียน label ใหม่
- รูปที่ไม่มีกล่อง = negative จะถูกเก็บไว้เทรนด้วย — ช่วยลด false positive จากของสีเหลืองหลอก
- ถ้าไม่ใส่ `--names` จะตั้งชื่อชั่วคราว `class0...` ให้ (แก้ใน `data.yaml` ก่อนเทรนได้)

### 4) เทรน
```bash
python train.py --imgsz 416 --epochs 120 --batch 32
# RTX 5060 Ti (16GB): ลอง --batch 64 ได้; ถ้า CUDA out of memory ค่อยลดลง
```
ผลลัพธ์: `runs/detect/tactile/weights/best.pt`

### 5) Export ไปลง OAK
```bash
python scripts/export_blob.py --weights runs/detect/tactile/weights/best.pt --imgsz 416
```
แล้วอัปโหลด `best.pt` ที่ **https://tools.luxonis.com** เลือก target **RVC2**
→ ได้ `.blob` + JSON config สำหรับ DepthAI

---

## โครงสร้าง
```
tactile_detect/
├── setup_env.sh              # สร้าง venv + ติดตั้ง
├── requirements.txt
├── train.py                  # เทรน YOLO
├── scripts/
│   ├── import_ciracore.py    # CiRA CORE export -> dataset + data.yaml
│   └── export_blob.py        # best.pt -> ONNX (-> blob ที่ tools.luxonis.com)
├── raw_ciracore/             # << วางผล export จาก CiRA CORE ที่นี่
├── dataset/                  # (สร้างอัตโนมัติ)
└── data.yaml                 # (สร้างอัตโนมัติ)
```

## เช็กก่อนเทรน
- เก็บภาพ **เต็มเฟรมจากมุมติดตั้งจริงของ OAK** ไม่ใช่ภาพ crop กระเบื้องเดี่ยว ๆ
- ใส่ภาพ negative (มีของเหลืองหลอกแต่ไม่มี tactile) เพื่อลด false positive
- คุมความหลากหลายของแสง/พื้น/สถานที่ ให้ครบ
# tractile_model
# tractile_model
