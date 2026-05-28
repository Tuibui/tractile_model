#!/usr/bin/env python3
"""ตรวจจับ "ทางเดิน tactile สีเหลือง" ด้วย logic สี + รูปทรง + ตำแหน่งพื้น (ไม่ใช้ deep learning)

ตรรกะ (ตรวจสด ๆ ทีละเฟรม ไม่มี state):
  1. mask สีเหลือง (HSV) เฉพาะแถบล่าง bottom ของภาพ
  2. หา region ที่ "เหมือนทางบนพื้น" ที่สุด (อยู่บนพื้น + เรียวสูง + อยู่กลางภาพแถวเท้าเรา)
  3. ลากเส้นกึ่งกลาง (centerline) -> บอกทิศ ตรง/เลี้ยวซ้าย/เลี้ยวขวา + reach (%)

คืน 2 ภาพ: (ภาพปกติ+overlay, mask สีเหลือง)

ตัวอย่าง:
  python scripts/color_path_detect.py                                          # ดูสด data_3 (ค่าเริ่มต้น)
  python scripts/color_path_detect.py --src recordings/data_1/video.mp4        # ดูสดไฟล์อื่น
  python scripts/color_path_detect.py --src recordings/data_1/video.mp4 --no-live --out runs/cp --stride 5
  python scripts/color_path_detect.py --src recordings/data_1/frames --no-live --out runs/cp --limit 80
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def yellow_mask(bgr, h_lo, h_hi, s_lo, v_lo):
    """คืน mask ไบนารีของพิกเซลสีเหลือง (ทำความสะอาดด้วย morphology แล้ว)"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, 255, 255))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17))   # สูง = เชื่อมเส้นประแนวตั้ง
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)    # ลบจุดเล็ก ๆ
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)  # ต่อ tactile ที่ขาดเป็นช่วง
    mask = cv2.medianBlur(mask, 5)                                         # ลบ speckle ที่กะพริบเฟรมต่อเฟรม
    return mask


def score_region(c, W, H, horizon_y):
    """ให้คะแนน contour ว่า "เหมือน path บนพื้นแค่ไหน". คืน (ok, score, info)
    ok=False = ถูกตัดทิ้ง (เก็บไว้วาดสีแดงเพื่อโชว์ว่าทำไมไม่เอา)"""
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] else x + w / 2
    cy = M["m01"] / M["m00"] if M["m00"] else y + h / 2
    frame_area = W * H

    info = dict(box=(x, y, w, h), cxy=(int(cx), int(cy)), area=area)

    # --- ตัวกรองแข็ง (เข้าเงื่อนไขไหนไม่ผ่าน = ตัดทิ้ง) ---
    if area < 0.0008 * frame_area:                 # เล็กเกินไป = noise
        return False, 0.0, {**info, "why": "tiny"}
    if cy < horizon_y:                              # ศูนย์กลางอยู่เหนือ horizon = ไม่ใช่พื้น
        return False, 0.0, {**info, "why": "above-ground"}
    reaches_bottom = (y + h) > 0.80 * H             # ต่อถึงช่วงล่างของภาพ (ที่เท้าเรา)
    elong = h / max(w, 1)                           # เรียวสูง = แถบทางเดิน

    # x ที่ "ขอบล่างสุด" ของ region — path ที่เราเดินอยู่ต้องอยู่กลางภาพแถวเท้าเรา
    pts = c.reshape(-1, 2)
    ymax = pts[:, 1].max()
    bottom_cx = pts[pts[:, 1] >= ymax - 8][:, 0].mean()
    center_off = abs(bottom_cx - W / 2) / (W / 2)   # 0=กลางพอดี, 1=ขอบ

    # --- คะแนน (ยิ่งสูงยิ่งเหมือน path) ---
    score = (area / frame_area)                     # ใหญ่ดีกว่า
    score *= (2.0 if reaches_bottom else 0.3)       # ต่อถึงพื้นล่าง = บวกหนัก
    score *= min(elong, 4.0) / 4.0 + 0.25           # แนวตั้งยาว = บวก
    score *= max(0.05, 1.0 - center_off)            # ขอบล่างยิ่งใกล้กลาง ยิ่งใช่ (ตัดแท็กซี่ข้าง ๆ)
    info.update(why="candidate", reaches_bottom=reaches_bottom,
                elong=round(elong, 2), center_off=round(center_off, 2))
    return True, score, info


def centerline(mask_region, y0, y1, step=8):
    """ไล่ทีละแถว หา x 'กลาง' (median กันค่าหลุด) ของพิกเซลเหลือง -> จุดกึ่งกลาง path"""
    pts = []
    for y in range(y0, y1, step):
        xs = np.where(mask_region[y] > 0)[0]
        if xs.size >= 3:
            pts.append((int(np.median(xs)), y))          # median เสถียรกว่า mean เมื่อมี blob ข้าง ๆ
    return pts


def fit_path(pts):
    """ฟิตเส้นตรง x = m*y + b ผ่านจุด centerline (least-squares) เพื่อ 'เฉลี่ย' jitter ทิ้ง
    คืน (m, b, y_top, y_bot) หรือ None ถ้าจุดน้อยเกินไป"""
    if len(pts) < 3:
        return None
    arr = np.asarray(pts, float)
    m, b = np.polyfit(arr[:, 1], arr[:, 0], 1)           # x เป็นฟังก์ชันของ y (รับเส้นเกือบตั้งได้)
    return m, b, int(arr[:, 1].min()), int(arr[:, 1].max())


def fit_points(fit, W, step=10):
    """แปลงผลฟิตเป็นจุดเส้นกึ่งกลางเรียบ ๆ (สำหรับวาด) ภายในช่วง y ที่ตรวจเจอจริง"""
    if fit is None:
        return []
    m, b, yt, yb = fit
    return [(int(np.clip(m * y + b, 0, W - 1)), y) for y in range(yt, yb + 1, step)]


def draw_nav_arrows(img, pts, color, thickness=2, arrow=15, step=24):
    """วาดลูกศรสั้น ๆ หลายอันเรียงต่อกันตามแนว path ชี้ทิศเดินหน้า (ใกล้เท้า -> ไกล)
    = HUD นำทางแบบ chevron ไม่ใช่แท่งยาวอันเดียว"""
    if len(pts) < 2:
        return
    pts = np.asarray(pts, np.float32)
    if pts[0][1] < pts[-1][1]:                   # ไล่จากใกล้เท้า(y มาก) -> ไกล(y น้อย) = เดินหน้า
        pts = pts[::-1]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return

    def at(d):                                   # จุดบนเส้นที่ระยะสะสม d
        i = min(max(int(np.searchsorted(cum, d) - 1), 0), len(seg) - 1)
        t = (d - cum[i]) / max(seg[i], 1e-6)
        p = pts[i] + t * (pts[i + 1] - pts[i])
        return int(round(p[0])), int(round(p[1]))

    d = 0.0
    while d + arrow <= total:
        cv2.arrowedLine(img, at(d), at(d + arrow), color, thickness, cv2.LINE_AA, tipLength=0.45)
        d += step


def heading_from_fit(fit, roi_y, H, dead_deg=10):
    """สรุปทิศจาก 'มุม' ของเส้นที่ฟิตเทียบกับเส้น ref กลาง (แนวตั้ง)
    คืน (heading, angle_deg) โดย angle +ขวา / -ซ้าย; |angle|<dead_deg = STRAIGHT"""
    if fit is None:
        return "??", 0.0
    m, _, yt, yb = fit
    look = H - roi_y
    if (yb - yt) < 0.35 * look:                  # path สั้นเกินไปแนวตั้ง = ความชันมั่ว -> อย่าเชื่อ
        return "STRAIGHT", 0.0
    angle = float(np.degrees(np.arctan(-m)))     # มุมจากแนวตั้ง: +ขวา / -ซ้าย
    angle = round(max(-89.0, min(89.0, angle)), 1)
    if abs(angle) < dead_deg:
        return "STRAIGHT", angle
    if angle < 0:
        return f"TURN LEFT {abs(angle):.0f}", angle
    return f"TURN RIGHT {angle:.0f}", angle


def process(bgr, args):
    """ตรวจทีละเฟรม. คืน (view, mask_vis, result)"""
    H, W = bgr.shape[:2]
    horizon_y = int(args.horizon * H)
    bottom_frac = getattr(args, "bottom", 0.50)
    roi_y = int((1.0 - bottom_frac) * H)            # ตรวจเฉพาะแถบล่าง bottom_frac ของภาพ
    mask = yellow_mask(bgr, args.h_lo, args.h_hi, args.s_lo, args.v_lo)
    mask[:roi_y] = 0                                # ตัดเหนือ ROI ทิ้ง (สนแค่ใกล้เท้า)
    center_w = getattr(args, "center", 0.40)
    cx_m = W // 2
    half_w = int(center_w * W / 2)                  # ครึ่งความกว้างของแถบกลาง
    mask[:, :cx_m - half_w] = 0                     # ตัดซ้ายแถบกลางทิ้ง
    mask[:, cx_m + half_w:] = 0                     # ตัดขวาแถบกลางทิ้ง
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ok_list, rejected = [], []
    for c in cnts:
        ok, score, info = score_region(c, W, H, horizon_y)
        if ok:
            ok_list.append((c, score, info))
        else:
            rejected.append(c)
    ok_list.sort(key=lambda x: -x[1])

    # STOP detection — เจอเงื่อนไขใดเงื่อนไขหนึ่งให้แจ้ง "STOP - DECIDE":
    #   (A) ทาง "strong" (ใหญ่+ถึงพื้น) ≥2 ทาง แยกข้างกัน ≥25% — Y-fork
    #   (B) แถบยาวแนวนอน (w/h > 2.5, กว้าง > 30%) — ทางตัด/T-junction/แถบเตือนหยุด
    stop_contours = []
    strong = [(c, info) for c, _, info in ok_list
              if info["area"] > 0.005 * H * W and info["reaches_bottom"]]
    if len(strong) >= 2:
        cxs = sorted(info["cxy"][0] for _, info in strong[:3])
        if cxs[-1] - cxs[0] > 0.25 * W:
            stop_contours.extend(c for c, _ in strong[:3])
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if (by >= roi_y and cv2.contourArea(c) > 0.004 * H * W
                and bw > 0.30 * W and bw / max(bh, 1) > 2.5):
            stop_contours.append(c)
            break                                # หนึ่งแถบก็พอสำหรับกรณี (B)
    is_fork = len(stop_contours) > 0

    out = bgr.copy()
    cv2.line(out, (cx_m - half_w, roi_y), (cx_m - half_w, H), (255, 0, 255), 2)  # ขอบซ้ายแถบกลาง
    cv2.line(out, (cx_m + half_w, roi_y), (cx_m + half_w, H), (255, 0, 255), 2)  # ขอบขวาแถบกลาง
    cv2.line(out, (cx_m - half_w, roi_y), (cx_m + half_w, roi_y), (255, 0, 255), 2)  # ขอบบน ROI
    if getattr(args, "show_rejected", False):
        for c in rejected:
            cv2.drawContours(out, [c], -1, (0, 0, 255), 2)

    region = np.zeros(mask.shape, np.uint8)
    result = dict(found=False, heading="", reach=0.0, angle=0.0, fork=False)
    fit, line_pts = None, []

    if is_fork:
        # ระบายทุก contour ที่ trigger STOP เป็นส้ม + แจ้ง
        for c in stop_contours:
            cv2.drawContours(region, [c], -1, 255, -1)
        region = cv2.bitwise_and(region, mask)
        overlay = out.copy()
        overlay[region > 0] = (0, 140, 255)
        out = cv2.addWeighted(overlay, 0.40, out, 0.60, 0)
        for c in stop_contours:
            cv2.drawContours(out, [c], -1, (0, 110, 220), 3)
        top_y = min(cv2.boundingRect(c)[1] for c in stop_contours)
        result.update(found=True, fork=True, heading="STOP - DECIDE",
                      reach=round((H - top_y) / H * 100.0, 1))
        cv2.putText(out, "STOP - DECIDE  (FORK / CROSS)", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 140, 255), 2)
    elif ok_list:
        c, score, info = ok_list[0]
        x, y, w, h = info["box"]
        cv2.drawContours(region, [c], -1, 255, -1)
        region = cv2.bitwise_and(region, mask)
        overlay = out.copy()
        overlay[region > 0] = (0, 255, 0)           # path -> เขียวโปร่งใส
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        cv2.drawContours(out, [c], -1, (0, 180, 0), 3)

        pts = centerline(region, y, y + h)
        fit = fit_path(pts)                          # ฟิตเส้นตรง -> ใช้คำนวณมุมเทียบ ref grid
        line_pts = fit_points(fit, W)
        for p in pts:                                # เส้นน้ำเงิน = centerline จริงตามรูปทาง
            cv2.circle(out, p, 3, (255, 0, 0), -1)
        if len(pts) >= 2:
            cv2.polylines(out, [np.array(pts)], False, (255, 0, 0), 2)
        heading, angle = heading_from_fit(fit, roi_y, H)

        top_y = fit[2] if fit is not None else y
        reach = (H - top_y) / H * 100.0
        result.update(found=True, heading=heading, reach=round(reach, 1), angle=angle)
        cv2.putText(out, f"PATH  {heading}  ({angle:+.0f} deg)  reach~{reach:.0f}%",
                    (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 0), 2)
    else:
        cv2.putText(out, "NO PATH", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # ---------- หน้าต่าง 2: mask + detection & navigation ----------
    mask_vis = np.zeros((H, W, 3), np.uint8)
    mask_vis[mask > 0] = (0, 120, 120)             # เหลืองดิบที่เจอ (จาง)
    mask_vis[region > 0] = (0, 255, 255)           # detection: ทางที่ตรวจเจอ (เหลืองสด)
    cv2.line(mask_vis, (0, roi_y), (W, roi_y), (255, 0, 255), 2)          # ขอบ ROI
    cx_ref = W // 2
    cv2.line(mask_vis, (cx_ref, roi_y), (cx_ref, H), (255, 255, 255), 1)  # navigation: เส้น ref กลาง
    if result.get("fork"):
        mask_vis[region > 0] = (0, 140, 255)       # ทางแยก -> ส้ม
        cv2.putText(mask_vis, "STOP - DECIDE  (FORK)", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
    elif result["found"] and len(line_pts) >= 2:
        near = line_pts[-1]                        # จุดใกล้เท้า (y มากสุด)
        draw_nav_arrows(mask_vis, line_pts, (0, 165, 255), 3)
        off = (near[0] - cx_ref) / (W / 2)         # ทางเยื้องซ้าย/ขวาจากกลาง (-ซ้าย/+ขวา)
        nav = f"{result['heading']}  ({result['angle']:+.0f} deg)  off={off:+.2f}"
        cv2.putText(mask_vis, nav, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    else:
        cv2.putText(mask_vis, "NO PATH", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out, mask_vis, result


def write_summary(out, rows):
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "path_found", "heading", "reach_pct", "angle_deg", "fork"])
        w.writerows(rows)


def run_images(src, out, args):
    """โหมดโฟลเดอร์รูป: เขียนภาพ view + mask ทีละไฟล์"""
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.only:
        files = [p for p in files if p.name == args.only]
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][:args.limit]
    if not files:
        raise SystemExit(f"ไม่พบรูปใน {src}")

    (out / "mask").mkdir(exist_ok=True)
    rows, n_found = [], 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        view, mask_vis, r = process(img, args)
        cv2.imwrite(str(out / p.name), view)
        cv2.imwrite(str(out / "mask" / p.name), mask_vis)
        rows.append([p.name, r["found"], r["heading"], r["reach"], r["angle"], r["fork"]])
        n_found += int(r["found"])
    write_summary(out, rows)
    print(f"ประมวลผล {len(rows)} เฟรม | เจอ path {n_found} ({100*n_found/max(len(rows),1):.0f}%)")
    print(f"view: {out}/  | mask: {out}/mask/  | summary.csv")


def run_video(src, out, args):
    """โหมด mp4: เขียนวิดีโอ view+mask วางข้างกัน 1 ไฟล์ (+ summary.csv ต่อเฟรม)"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"เปิดวิดีโอไม่ได้: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = out / f"{src.stem}_annotated.mp4"
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    rows, n_found, idx, written = [], 0, -1, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % args.stride != 0:                 # ข้ามเฟรมตาม --stride (เร็วขึ้น)
            continue
        if args.limit and written >= args.limit:
            break
        view, _, r = process(frame, args)
        if writer is None:
            h, w = view.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        writer.write(view)
        rows.append([idx, r["found"], r["heading"], r["reach"], r["angle"], r["fork"]])
        n_found += int(r["found"])
        written += 1
        if written % 200 == 0:
            print(f"  ...{written} เฟรม (จากทั้งหมด ~{total // args.stride})")

    cap.release()
    if writer is not None:
        writer.release()
    write_summary(out, rows)
    print(f"ประมวลผล {written} เฟรม (stride {args.stride}) | เจอ path {n_found} "
          f"({100*n_found/max(written,1):.0f}%)")
    print(f"วิดีโอผล: {out_path}")
    print(f"summary.csv: {out}/summary.csv")


def run_live(src, args):
    """โหมดดูสด: เล่นวิดีโอ + overlay ผลตรวจจับแบบเรียลไทม์ (หน้าต่างเดียว)
    ปุ่ม:  q=ออก | space=หยุด/เล่น | a,d=ถอย/เดินหน้าทีละเฟรม (ตอนหยุด)
    Alert: throttle เหมือน blind_navigation — แจ้งเฉพาะเลี้ยว/STOP ทุก ๆ alert_interval วินาที"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"เปิดวิดีโอไม่ได้: {src}")
    delay = max(1, int(1000 / args.fps))
    win_v = "view  q=quit  space=pause  a/d=step"
    cv2.namedWindow(win_v, cv2.WINDOW_NORMAL)

    # alert gate (rate-limit) — แบบเดียวกับ CALL_INTERVAL ใน blind_navigation
    interval = getattr(args, "alert_interval", 3.0)
    banner_dur = 1.8
    last_alert_t, last_alert_lbl = 0.0, ""
    banner_until, banner_text, banner_col = 0.0, "", (0, 0, 255)

    paused, frame = False, None
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:                                  # จบวิดีโอ -> วนซ้ำ
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        view, _, r = process(frame, args)

        # --- alert gate: ใช้เฉพาะเลี้ยว/STOP, throttle เวลา + เปลี่ยน label ก็ปล่อยทันที ---
        lbl = r["heading"]
        now = time.time()
        actionable = lbl.startswith("TURN ") or lbl.startswith("STOP")
        if actionable and (lbl != last_alert_lbl or now - last_alert_t >= interval):
            print(f"[ALERT {time.strftime('%H:%M:%S')}] {lbl}")
            last_alert_t, last_alert_lbl = now, lbl
            banner_until = now + banner_dur
            banner_text = lbl
            banner_col = (0, 140, 255) if lbl.startswith("STOP") else (0, 0, 255)
        if now < banner_until:
            h2, w2 = view.shape[:2]
            cv2.rectangle(view, (0, h2 - 80), (w2, h2 - 20), (0, 0, 0), -1)
            cv2.putText(view, "ALERT: " + banner_text, (20, h2 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, banner_col, 3)

        cv2.imshow(win_v, view)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        elif paused and key == ord("d"):
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif paused and key == ord("a"):
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
            ok, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="recordings/data_3/video.mp4", help="โฟลเดอร์รูป หรือไฟล์วิดีโอ (.mp4 ฯลฯ)")
    ap.add_argument("--out", default="runs/colorpath", help="โฟลเดอร์ผลลัพธ์")
    ap.add_argument("--live", action=argparse.BooleanOptionalAction, default=True,
                    help="ดูสด 2 หน้าต่าง (ค่าเริ่มต้น); ใช้ --no-live เพื่อ export เป็นไฟล์แทน")
    ap.add_argument("--fps", type=int, default=30, help="[โหมด --live] ความเร็วเล่น")
    ap.add_argument("--alert-interval", type=float, default=3.0,
                    help="[โหมด --live] วินาทีขั้นต่ำระหว่างแจ้งเตือนซ้ำ (แบบ CALL_INTERVAL ของ blind_navigation)")
    ap.add_argument("--limit", type=int, default=0, help="จำกัดจำนวนเฟรม (0=ทั้งหมด)")
    ap.add_argument("--only", default="", help="[โหมดรูป] ประมวลผลไฟล์เดียว (ชื่อไฟล์)")
    ap.add_argument("--stride", type=int, default=1, help="[โหมดวิดีโอ] ประมวลผลทุก ๆ N เฟรม (มาก=เร็ว)")
    ap.add_argument("--show-rejected", action="store_true", help="วาดเส้นแดงรอบเหลืองที่ถูกตัดทิ้ง")
    # --- ช่วงสีเหลือง (OpenCV H 0-179) ---
    ap.add_argument("--h_lo", type=int, default=5)
    ap.add_argument("--h_hi", type=int, default=20, help="ต่ำกว่า ~35 = กันใบไม้เขียว; ทางลากถึง H~38 (จูนแล้ว)")
    ap.add_argument("--s_lo", type=int, default=30, help="ต่ำ=จับทางซีดได้เยอะ (ROI แคบกัน noise นอกกรอบให้แล้ว); จูนสุด: 15")
    ap.add_argument("--v_lo", type=int, default=95, help="ต่ำ=จับทางในเงาได้; แทบไม่มีผลตอน ROI แคบ; จูนสุด: 60")
    ap.add_argument("--horizon", type=float, default=0.30, help="ส่วนบนของภาพที่ถือว่า 'ไม่ใช่พื้น'")
    ap.add_argument("--bottom", type=float, default=0.3, help="ตรวจเฉพาะแถบล่าง N ของภาพ (0.50=ล่าง 50%%)")
    ap.add_argument("--center", type=float, default=0.40,
                    help="ตรวจเฉพาะแถบกลาง N ของความกว้าง (0.40=กว้าง 40%% รอบกลาง = ±20%%)")
    args = ap.parse_args()
    args.stride = max(1, args.stride)

    src = Path(args.src)
    if args.live:
        if not (src.is_file() and src.suffix.lower() in VIDEO_EXTS):
            raise SystemExit(f"--live ต้องใช้กับไฟล์วิดีโอ ({', '.join(VIDEO_EXTS)})")
        run_live(src, args)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if src.is_file() and src.suffix.lower() in VIDEO_EXTS:
        run_video(src, out, args)
    elif src.is_dir():
        run_images(src, out, args)
    else:
        raise SystemExit(f"--src ต้องเป็นโฟลเดอร์รูป หรือไฟล์วิดีโอ ({', '.join(VIDEO_EXTS)})")
    print("เขียว=path / น้ำเงิน=centerline / ชมพู=ขอบ ROI / หน้าต่าง mask=เหลือง")


if __name__ == "__main__":
    main()
