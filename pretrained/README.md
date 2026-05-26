# pretrained/

วาง weight สำหรับ **warm-start fine-tune** ที่นี่ (ดู README หลัก → ข้อ 4b)

ไฟล์ `*.pt` ในโฟลเดอร์นี้ถูก gitignore (ไม่ขึ้น git) — ต้องคัดลอกมาเองด้วย scp/USB เช่น:

```
~/blind_navigation/yolo/best.pt   ->   pretrained/blindnav_best_v8n.pt
```

แล้วเทรนด้วย:

```bash
python train.py --model pretrained/blindnav_best_v8n.pt --imgsz 416 --epochs 120 --batch 32
```
