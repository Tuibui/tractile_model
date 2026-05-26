#!/usr/bin/env python3
"""บันทึกจาก OAK-D-Lite (depthai 3.x) สำหรับเก็บ dataset tactile paving

แต่ละ session สร้างโฟลเดอร์ cap_YYYYMMDD_HHMMSS/ ที่มี (แยกไฟล์กัน):
  - video.mp4          : วิดีโอเต็ม ทุกเฟรมตาม --fps (ค่าเริ่มต้น 30fps) ไว้ดูย้อนหลัง/แยกเฟรมเพิ่ม
  - frames/frame_*.jpg : รูปนิ่ง แคปทุก ๆ --snap-sec วินาที (ค่าเริ่มต้น 1 วิ) ไว้เอาไปติดป้าย/เทรน
รูปนิ่งเขียนจากไบต์ JPEG ของกล้องตรง ๆ (on-device MJPEG q97) = คมเต็มคุณภาพ ไม่ encode ซ้ำ

- ค่าเริ่มต้น 720p + ล็อกโฟกัส (--focus) กันภาพเบลอตอนเดินเก็บ dataset
- พรีวิวสด, กด r เริ่ม/หยุด (หลาย session), q ออก
- พรีวิวมีตัววัดความคม SHARP (Laplacian variance) — เล็ง/ปรับ --focus ให้ค่าสูงสุดก่อนเดิน (>=100 ถือว่าคม)
- --autostart เริ่มทันที, --no-preview โหมด headless (พับจอเดินเก็บ)

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
    ap.add_argument("--out", default="recordings", help="โฟลเดอร์หลัก (สร้าง subfolder cap_<เวลา>/ ต่อ session)")
    ap.add_argument("--width", type=int, default=1280, help="ค่าเริ่มต้น 720p (1280x720)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30, help="เฟรมเรตวิดีโอ (ถ้า USB หลุด/X_LINK ลองลดเป็น 15)")
    ap.add_argument("--snap-sec", type=float, default=1.0,
                    help="แคปรูปนิ่งทุกกี่วินาที (ค่าเริ่มต้น 1.0); ใส่ 0 = ทุกเฟรม")
    ap.add_argument("--focus", type=int, default=130,
                    help="ล็อกโฟกัส 0-255 (มาก=โฟกัสใกล้); -1 = ออโต้โฟกัส. "
                         "ล็อกไว้กันภาพเบลอตอนเดิน — ปรับให้ค่า SHARP ในพรีวิวสูงสุด")
    ap.add_argument("--quality", type=int, default=97, help="คุณภาพ MJPEG 1-100 (สูง=ชัด/ไฟล์ใหญ่)")
    ap.add_argument("--orientation", choices=list(ORIENTATIONS), default="normal",
                    help="rot180 ถ้าติดกล้องกลับหัว")
    ap.add_argument("--no-preview", action="store_true",
                    help="โหมด headless: แคปทันที หยุดด้วย Ctrl+C")
    ap.add_argument("--autostart", "-a", action="store_true",
                    help="เริ่มแคปทันทีตอนรัน (ไม่ต้องกด r); ยังมีพรีวิว กด r หยุด/เริ่มใหม่ได้")
    args = ap.parse_args()

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    with dai.Pipeline() as pipe:
        cam = pipe.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        cam.setImageOrientation(ORIENTATIONS[args.orientation])

        # ล็อกโฟกัส: ตอนเดินเก็บ dataset ออโต้โฟกัสจะ "วิ่งหา" โฟกัสตลอด -> ภาพเบลอ
        # ล็อกไว้ที่ค่าเดียวให้คมที่ระยะพื้น/กระเบื้องด้านหน้า (~1-2 ม.) แล้วไม่ขยับอีก
        if args.focus is not None and args.focus >= 0:
            try:
                cam.initialControl.setManualFocus(int(args.focus))
                print(f"[FOCUS] ล็อกโฟกัสที่ {args.focus} (0-255)")
            except Exception as e:
                print(f"[FOCUS] ⚠️ ตั้งโฟกัสล็อกไม่สำเร็จ ({e}) -> ใช้ออโต้โฟกัสแทน")
        else:
            print("[FOCUS] ออโต้โฟกัส (AF) — อาจเบลอตอนเดิน")

        cam_out = cam.requestOutput((args.width, args.height), dai.ImgFrame.Type.NV12, fps=args.fps)

        enc = pipe.create(dai.node.VideoEncoder)
        enc.setDefaultProfilePreset(args.fps, dai.VideoEncoderProperties.Profile.MJPEG)
        enc.setQuality(args.quality)
        cam_out.link(enc.input)

        q = enc.bitstream.createOutputQueue(maxSize=30, blocking=False)
        pipe.start()

        st = {"dir": None, "video": None, "vpath": None, "start": 0.0,
              "snaps": 0, "last_snap": 0.0, "sessions": 0, "total_snaps": 0}
        recording = args.no_preview or args.autostart

        def open_session():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sdir = out_dir / f"cap_{ts}"
            (sdir / "frames").mkdir(parents=True, exist_ok=True)
            st["dir"] = sdir
            st["vpath"] = sdir / "video.mp4"
            st["video"] = None        # สร้างตอนได้เฟรมแรก (รู้ขนาดจริง)
            st["start"] = time.time()
            st["last_snap"] = 0.0     # 0 = แคปรูปแรกทันที
            st["snaps"] = 0
            st["sessions"] += 1
            print(f"[REC] เริ่ม -> {sdir}/  (video.mp4 @ {args.fps}fps + frames/ ทุก {args.snap_sec:g}s)")

        def close_session():
            if st["dir"] is not None:
                if st["video"] is not None:
                    st["video"].release()
                dt = time.time() - st["start"]
                print(f"[REC] หยุด ({dt:.1f}s | วิดีโอ + {st['snaps']} รูป) -> {st['dir']}/")
                st["dir"] = None
                st["video"] = None

        try:
            while pipe.isRunning():
                pkt = q.get()
                raw = pkt.getData()  # ไบต์ JPEG ที่กล้องบีบอัดมาแล้ว (on-device MJPEG)

                if recording and st["dir"] is None:
                    open_session()
                elif not recording and st["dir"] is not None:
                    close_session()

                frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                if st["dir"] is not None:
                    # วิดีโอ: เขียนทุกเฟรม (เต็ม fps)
                    if st["video"] is None:
                        h, w = frame.shape[:2]
                        st["video"] = cv2.VideoWriter(str(st["vpath"]), fourcc, args.fps, (w, h))
                    st["video"].write(frame)
                    # รูปนิ่ง: เขียนไบต์ JPEG ตรง ๆ ทุก ๆ snap_sec วินาที (คมเต็มคุณภาพ, ไฟล์แยก)
                    now = time.time()
                    if now - st["last_snap"] >= args.snap_sec:
                        (st["dir"] / "frames" / f"frame_{st['snaps']:06d}.jpg").write_bytes(bytes(raw))
                        st["snaps"] += 1
                        st["total_snaps"] += 1
                        st["last_snap"] = now

                if args.no_preview:
                    continue

                disp = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
                if recording:
                    cv2.circle(disp, (22, 22), 8, (0, 0, 255), -1)
                    cv2.putText(disp, f"REC {time.time() - st['start']:4.0f}s | {st['snaps']} imgs", (38, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(disp, "press 'r' = rec, 'q' = quit", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # ตัววัดความคม: Laplacian variance ของทั้งเฟรม (เกณฑ์เดียวกับ data cleaning ~100)
                # ปรับ --focus ให้ค่านี้สูงสุด = โฟกัสคมที่สุด; ถ้าต่ำกว่า 100 = เสี่ยงเบลอ
                sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                sc = (0, 255, 0) if sharp >= 100 else (0, 165, 255)
                foc = "AF" if (args.focus is None or args.focus < 0) else str(args.focus)
                cv2.putText(disp, f"SHARP {sharp:4.0f}  focus={foc}", (12, disp.shape[0] - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
                cv2.imshow("OAK-D-Lite | tactile recorder", disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    recording = not recording
        except KeyboardInterrupt:
            print("\nหยุดด้วย Ctrl+C")
        finally:
            close_session()
            cv2.destroyAllWindows()
            print(f"เสร็จ: {st['sessions']} session, รวม {st['total_snaps']} รูป (+ วิดีโอ) ใน {out_dir}")


if __name__ == "__main__":
    main()
