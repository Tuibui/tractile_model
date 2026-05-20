#!/usr/bin/env python3
"""Export weights ที่เทรนแล้ว -> ONNX เพื่อนำไปแปลงเป็น .blob สำหรับ OAK-D-Lite (RVC2)

เส้นทางแปลงเป็น .blob (เลือกอย่างใดอย่างหนึ่ง):
  1) ง่ายสุด + แนะนำ: อัปโหลด best.pt ที่ https://tools.luxonis.com
     -> เลือก target "RVC2" -> ได้ทั้ง .blob และไฟล์ JSON config สำหรับ DepthAI
     (เครื่องมือนี้รู้วิธีถอดหัว YOLO ให้ตรงกับ YoloDetectionNetwork ของ DepthAI)
  2) ทำเองในเครื่อง: ONNX (จากสคริปต์นี้) -> OpenVINO IR -> blob ด้วย `blobconverter`

ตัวอย่าง:
  python scripts/export_blob.py --weights runs/detect/tactile/weights/best.pt --imgsz 416
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=416, help="ต้องตรงกับตอนเทรน และหารด้วย 32 ลงตัว")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    model = YOLO(args.weights)
    path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True)
    print(f"\nได้ ONNX: {path}")
    print("ขั้นต่อไป -> อัปโหลด best.pt หรือ ONNX นี้ที่ https://tools.luxonis.com (เลือก RVC2)")
    print("จะได้ .blob + JSON config ไปใช้กับ YoloSpatialDetectionNetwork ใน DepthAI")


if __name__ == "__main__":
    main()
