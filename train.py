#!/usr/bin/env python3
"""เทรน YOLO detection สำหรับ tactile paving เพื่อรันบน OAK-D-Lite (RVC2)

จำนวนคลาสมาจาก data.yaml อัตโนมัติ (ตอนนี้คลาสเดียว: tactile)

ตัวอย่าง:
  python train.py                         # ค่าเริ่มต้น (yolo11n, imgsz 416)
  python train.py --epochs 150 --batch 8  # ถ้า CUDA out-of-memory ให้ลด batch
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt / yolov8n.pt (nano = เบา รันบน OAK ได้)")
    ap.add_argument("--imgsz", type=int, default=416, help="416 เหมาะกับ Myriad X; ต้องหารด้วย 32 ลงตัว")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32, help="RTX 5060 Ti 16GB: ลองได้ถึง 64; ถ้า OOM ค่อยลด")
    ap.add_argument("--device", default="0", help="0 = GPU, cpu = ใช้ CPU")
    ap.add_argument("--name", default="tactile")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=30,          # early stop ถ้าไม่ดีขึ้น 30 epoch
        cache=False,          # ตั้งเป็น "ram" ได้ถ้า RAM พอ เพื่อเทรนเร็วขึ้น
        # --- augmentation เน้นแก้ปัญหาแสง/กระเบื้องสีซีด ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,            # ความสว่างแปรผัน (แดด/เงา/แสงน้อย)
        translate=0.1,
        scale=0.5,            # กระเบื้องใกล้-ไกล ขนาดต่างกัน
        fliplr=0.5,
        mosaic=1.0,
    )
    print("\nเสร็จ! ผลอยู่ใน runs/detect/%s/  ->  weights/best.pt" % args.name)
    print("ต่อไป: python scripts/export_blob.py --weights runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
