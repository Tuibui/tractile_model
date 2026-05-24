#!/usr/bin/env python3
"""บันทึกวิดีโอจาก OAK-D-Lite (depthai 3.x) สำหรับเก็บ dataset tactile paving

ใช้ **on-device MJPEG encoding**: OAK บีบอัดบนตัวกล้องก่อนส่ง USB (~5-10 MB/s)
-> เสถียรบน USB2 (สตรีมเฟรมดิบ 1080p ~90 MB/s จะล้น USB2 แล้ว crash)
host รับ JPEG -> decode -> เขียนเป็น .mp4 (เล่นได้เลย ไม่ต้องพึ่ง ffmpeg)

- พรีวิวสด, กด r เริ่ม/หยุดอัด (อัดได้หลายคลิป), q ออก
- --autostart เริ่มอัดทันที, --no-preview โหมด headless (พับจอเดินเก็บ)

depthai ไม่ได้อยู่ใน requirements.txt (เครื่องเทรนไม่ต้องใช้) — รันบนเครื่องที่ต่อ OAK:
  /home/tuibui/collect_image/venv/bin/python scripts/record_oak.py --out recordings
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai

ORIENTATIONS = {
    "normal": dai.CameraImageOrientation.NORMAL,
    "rot180": dai.CameraImageOrientation.ROTATE_180_DEG,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="recordings", help="โฟลเดอร์เก็บวิดีโอ")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=97, help="คุณภาพ MJPEG 1-100 (สูง=ชัด/ไฟล์ใหญ่)")
    ap.add_argument("--orientation", choices=list(ORIENTATIONS), default="normal",
                    help="rot180 ถ้าติดกล้องกลับหัว")
    ap.add_argument("--no-preview", action="store_true",
                    help="โหมด headless: อัดทันที หยุดด้วย Ctrl+C")
    ap.add_argument("--autostart", "-a", action="store_true",
                    help="เริ่มอัดทันทีตอนรัน (ไม่ต้องกด r); ยังมีพรีวิว กด r หยุด/เริ่มใหม่ได้")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with dai.Pipeline() as pipe:
        cam = pipe.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam.setImageOrientation(ORIENTATIONS[args.orientation])
        cam_out = cam.requestOutput((args.width, args.height), dai.ImgFrame.Type.NV12, fps=args.fps)

        enc = pipe.create(dai.node.VideoEncoder)
        enc.setDefaultProfilePreset(args.fps, dai.VideoEncoderProperties.Profile.MJPEG)
        enc.setQuality(args.quality)
        cam_out.link(enc.input)

        q = enc.bitstream.createOutputQueue(maxSize=30, blocking=False)
        pipe.start()

        st = {"writer": None, "path": None, "start": 0.0, "clips": 0}
        recording = args.no_preview or args.autostart

        def open_writer(frame):
            h, w = frame.shape[:2]
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st["path"] = out_dir / f"tactile_{ts}.mp4"
            st["writer"] = cv2.VideoWriter(str(st["path"]), fourcc, args.fps, (w, h))
            st["start"] = time.time()
            st["clips"] += 1
            print(f"[REC] เริ่มอัด -> {st['path']}")

        def close_writer():
            if st["writer"] is not None:
                st["writer"].release()
                print(f"[REC] หยุด ({time.time() - st['start']:.1f}s) -> {st['path']}")
                st["writer"] = None

        try:
            while pipe.isRunning():
                pkt = q.get()
                frame = cv2.imdecode(pkt.getData(), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                if recording and st["writer"] is None:
                    open_writer(frame)
                elif not recording and st["writer"] is not None:
                    close_writer()
                if st["writer"] is not None:
                    st["writer"].write(frame)

                if args.no_preview:
                    continue

                disp = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                if recording:
                    cv2.circle(disp, (22, 22), 8, (0, 0, 255), -1)
                    cv2.putText(disp, f"REC {time.time() - st['start']:5.1f}s", (38, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(disp, "press 'r' = record, 'q' = quit", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("OAK-D-Lite | tactile recorder", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
        except KeyboardInterrupt:
            print("\nหยุดด้วย Ctrl+C")
        finally:
            close_writer()
            cv2.destroyAllWindows()
            print(f"เสร็จ: อัด {st['clips']} คลิป ใน {out_dir}")


if __name__ == "__main__":
    main()
