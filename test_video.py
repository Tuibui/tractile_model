#!/usr/bin/env python3
"""ทดสอบโมเดล best.pt กับวิดีโอ mp4 ใน recordings/ แล้วแสดงผลแบบเรียลไทม์ (ไม่บันทึกไฟล์)

ตัวอย่าง:
  python test_video.py                              # รันทุก mp4 ใน recordings/
  python test_video.py recordings/data_1/video.mp4  # รันไฟล์เดียว
  python test_video.py --conf 0.4 --device 0        # ปรับ confidence / ใช้ GPU

ปุ่มลัดระหว่างเล่น:
  q / ESC = ออก,  n = วิดีโอถัดไป,  space = หยุดชั่วคราว/เล่นต่อ
"""
import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "tactile-2" / "weights" / "best.pt"
DEFAULT_DIR = ROOT / "recordings"
WIN = "tactile_detect (q=ออก  n=ถัดไป  space=หยุด)"


def find_videos(paths):
    """รับ path เป็นไฟล์หรือโฟลเดอร์ คืน list ของไฟล์ mp4 ที่เรียงแล้ว"""
    vids = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            vids += sorted(p.rglob("*.mp4"))
        elif p.is_file():
            vids.append(p)
        else:
            print(f"ข้าม (ไม่พบ): {p}", file=sys.stderr)
    return vids


def play(model, video, conf, imgsz, device, scale):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"เปิดไม่ได้: {video}", file=sys.stderr)
        return True  # ไปวิดีโอถัดไป

    print(f"กำลังเล่น: {video}")
    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break  # จบวิดีโอ
            res = model.predict(frame, conf=conf, imgsz=imgsz,
                                device=device, verbose=False)[0]
            annotated = res.plot()
            n = len(res.boxes)
            cv2.putText(annotated, f"{video.parent.name}/{video.name}  det={n}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)
            if scale != 1.0:
                annotated = cv2.resize(annotated, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WIN, annotated)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key in (ord("q"), 27):       # q หรือ ESC
            cap.release()
            return False                # ขอออกทั้งหมด
        if key == ord("n"):             # ข้ามไปวิดีโอถัดไป
            break
        if key == ord(" "):             # หยุด/เล่นต่อ
            paused = not paused

    cap.release()
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", default=[str(DEFAULT_DIR)],
                    help="ไฟล์ mp4 หรือโฟลเดอร์ (ค่าเริ่มต้น: recordings/)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="พาธ best.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=416, help="ขนาดอินพุต (ตอนเทรนใช้ 416)")
    ap.add_argument("--device", default="cpu", help="cpu = ใช้ CPU, 0 = GPU")
    ap.add_argument("--scale", type=float, default=1.0, help="ขยายภาพที่แสดง เช่น 1.5 = ใหญ่ขึ้น 1.5 เท่า")
    ap.add_argument("--fullscreen", action="store_true", help="แสดงแบบเต็มจอ")
    args = ap.parse_args()

    videos = find_videos(args.sources)
    if not videos:
        print("ไม่พบไฟล์ mp4", file=sys.stderr)
        sys.exit(1)

    print(f"โหลดโมเดล: {args.weights}")
    model = YOLO(args.weights)

    # หน้าต่างปรับขนาดได้ + ลากขยายเองได้
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN, 1280, 720)

    for v in videos:
        keep_going = play(model, v, args.conf, args.imgsz, args.device, args.scale)
        if not keep_going:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
