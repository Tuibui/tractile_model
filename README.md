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

### 4b) (ทางเลือก) fine-tune จากโมเดลที่เทรน tactile paving มาแล้ว — warm-start
แทนที่จะเริ่มจาก `yolo11n.pt` (COCO) สามารถ warm-start จาก weight ที่รู้จัก tactile paving อยู่แล้ว
เช่น `best.pt` จากโปรเจกต์ blind_navigation (**YOLOv8n, 2 คลาส**) — เหมาะเมื่อ dataset เล็ก
```bash
# คัดลอก best.pt มาวางเอง (scp/USB) — *.pt ถูก gitignore จึงอยู่เฉพาะเครื่อง ไม่ขึ้น repo
mkdir -p pretrained
# ต้นทาง: ~/blind_navigation/yolo/best.pt   ->   pretrained/blindnav_best_v8n.pt

python train.py --model pretrained/blindnav_best_v8n.pt --imgsz 416 --epochs 120 --batch 32
```
- **ได้โมเดล YOLOv8n** (สถาปัตยกรรมตามไฟล์ checkpoint ไม่ใช่ v11n) — export `.blob` RVC2 ได้ปกติ
- จำนวนคลาสต่างกัน (2→1): Ultralytics **รีเซ็ตหัว detection เป็น 1 คลาสอัตโนมัติ** แต่เก็บ backbone+neck เดิม
  (จะเห็น log `Transferred .../... items`) = ข้อดีของ warm-start
- warm-start จาก v8 ข้ามไป v11 ไม่ได้ (อาร์คต่างกัน); ถ้าต้องการ v11n ให้เทรนสดจาก `yolo11n.pt` (ข้อ 4)

> ⚠️ **ระวัง `best.pt` ซ้ำชื่อ:** ต้นทาง = `pretrained/blindnav_best_v8n.pt` (2 คลาส),
> ผลลัพธ์ = `runs/detect/tactile/weights/best.pt` (1 คลาส `tactile` — ตัวนี้เอาไป export ข้อ 5)

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
├── pretrained/               # (ออปชัน, *.pt ไม่ขึ้น git) weight warm-start เช่น best.pt
├── dataset/                  # (สร้างอัตโนมัติ)
└── data.yaml                 # (สร้างอัตโนมัติ)
```

## เช็กก่อนเทรน
- เก็บภาพ **เต็มเฟรมจากมุมติดตั้งจริงของ OAK** ไม่ใช่ภาพ crop กระเบื้องเดี่ยว ๆ
- ใส่ภาพ negative (มีของเหลืองหลอกแต่ไม่มี tactile) เพื่อลด false positive
- คุมความหลากหลายของแสง/พื้น/สถานที่ ให้ครบ
