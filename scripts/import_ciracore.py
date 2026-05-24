#!/usr/bin/env python3
"""แปลงผลลัพธ์จาก CiRA CORE (YOLO/Darknet export) -> โครงสร้าง dataset ของ Ultralytics

CiRA CORE export ออกมาเป็น: รูป + ไฟล์ .txt ต่อ 1 รูป (รูปแบบ YOLO: `class_id cx cy w h`
ค่า normalize 0-1)

⚠️  จุดสำคัญ: CiRA มักใส่ `class_id` เป็น "เลข global ของโปรเจกต์" (เช่น 451) ไม่ใช่ 0-based
และ obj.names ก็มักไม่ตรงกับ id ที่ใช้จริง (เช่นบอกว่ามี 2 คลาสชื่อ "0"/"1" แต่ label ใช้ id เดียว)
Ultralytics ต้องการ class id ที่ต่อเนื่อง 0..nc-1 ถ้าปล่อย id ดิบ (เช่น 451) ไว้ การเทรนจะ error ทันที

สคริปต์นี้จะ:
  - สแกนทุก label เพื่อหา class id ที่ "ถูกใช้จริง"
  - รีแมป id เหล่านั้นให้เป็น 0..k-1 ตามลำดับจากน้อยไปมาก (เช่น {451: 0})
  - เขียน label ใหม่ด้วย id ที่รีแมปแล้ว (จึง copy ไม่ใช่ symlink สำหรับ label)
  - แบ่ง train/val แล้วคัดลอก/ลิงก์รูปเข้า dataset/images/{train,val}
  - รูปที่ไม่มี .txt หรือ .txt ว่าง = negative -> เก็บไว้เทรนด้วย (ลด false positive)
  - เขียน data.yaml ให้อัตโนมัติ

ตัวอย่าง:
  python scripts/import_ciracore.py --src raw_ciracore --out dataset --val 0.2 --names tactile
  # ถ้ามีหลายคลาส ใส่ชื่อตามลำดับ id ที่ใช้จริง (น้อย->มาก): --names bar_tile dot_tile
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def label_for(img: Path) -> Path:
    """หาไฟล์ .txt ของรูป: ลองในโฟลเดอร์เดียวกันก่อน แล้วค่อยลอง labels/"""
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    sibling = img.parent.parent / "labels" / (img.stem + ".txt")
    return sibling if sibling.exists() else same  # คืน path เดิมแม้ไม่มี (= negative)


def read_boxes(lbl: Path):
    """อ่าน label -> list ของ (class_id:int, rest:str). ข้ามบรรทัดว่าง/ผิดรูป"""
    boxes = []
    if not (lbl.exists() and lbl.stat().st_size > 0):
        return boxes
    for ln in lbl.read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            boxes.append((int(float(parts[0])), " ".join(parts[1:])))
    return boxes


def scan_used_ids(images):
    """สแกนทุก label เพื่อรวบรวม class id ที่ถูกใช้จริง"""
    used = set()
    for img in images:
        for cid, _ in read_boxes(label_for(img)):
            used.add(cid)
    return sorted(used)


def place(files, split, out: Path, remap: dict, do_copy: bool):
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_neg = 0
    for img in files:
        # รูป: symlink (ค่าเริ่มต้น) หรือ copy — เนื้อไฟล์ไม่เปลี่ยน
        dst_img = img_dir / img.name
        if dst_img.exists() or dst_img.is_symlink():
            dst_img.unlink()
        if do_copy:
            shutil.copy2(img, dst_img)
        else:
            dst_img.symlink_to(img.resolve())
        # label: เขียนใหม่เสมอ เพราะต้องรีแมป class id
        boxes = read_boxes(label_for(img))
        dst_lbl = lbl_dir / (img.stem + ".txt")
        if boxes:
            dst_lbl.write_text("".join(f"{remap[c]} {rest}\n" for c, rest in boxes))
        else:
            dst_lbl.write_text("")  # negative: label ว่าง
            n_neg += 1
    return n_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="โฟลเดอร์ export จาก CiRA CORE")
    ap.add_argument("--out", default="dataset", help="โฟลเดอร์ dataset ปลายทาง")
    ap.add_argument("--val", type=float, default=0.2, help="สัดส่วน validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="คัดลอกรูป (ค่าเริ่มต้น = symlink)")
    ap.add_argument(
        "--names",
        nargs="*",
        help="ชื่อคลาสเรียงตาม id ที่ใช้จริง (น้อย->มาก). 1 คลาส: --names tactile",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"ไม่พบโฟลเดอร์ source: {src}")

    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"ไม่พบรูปใน {src}")

    used = scan_used_ids(images)
    if not used:
        raise SystemExit("ไม่พบกล่องใน label เลย (ทุกไฟล์ว่างหรือไม่มี .txt) — ตรวจ --src อีกที")
    remap = {old: new for new, old in enumerate(used)}
    print(f"class id ที่ใช้จริงใน label: {used}")
    if used != list(range(len(used))):
        print(f"รีแมปเป็น 0-based: {remap}")

    names = args.names
    if not names:
        names = [f"class{i}" for i in range(len(used))]
        print(f"⚠️  ไม่ได้ระบุ --names — ใช้ชื่อชั่วคราว {names} (แก้ใน data.yaml ก่อนเทรนได้)")
    if len(names) != len(used):
        raise SystemExit(
            f"จำนวน --names ({len(names)}) ไม่ตรงกับจำนวนคลาสที่ใช้จริง ({len(used)})\n"
            f"id ที่ใช้จริง: {used}  ->  ใส่ชื่อให้ครบตามลำดับนี้"
        )
    print(f"คลาส ({len(names)}): {names}")

    random.seed(args.seed)
    random.shuffle(images)
    n_val = int(len(images) * args.val)
    val_files, train_files = images[:n_val], images[n_val:]

    if out.exists():
        shutil.rmtree(out)
    neg_tr = place(train_files, "train", out, remap, args.copy)
    neg_va = place(val_files, "val", out, remap, args.copy)

    yaml_path = Path("data.yaml").resolve()
    lines = [
        f"path: {out}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {i}: {n}" for i, n in enumerate(names)],
        "",
    ]
    yaml_path.write_text("\n".join(lines))

    print(f"\nรวม {len(images)} รูป  ->  train {len(train_files)} / val {len(val_files)}")
    print(f"negative (ไม่มีกล่อง): train {neg_tr} / val {neg_va}")
    print(f"เขียน data.yaml -> {yaml_path}  (nc={len(names)})")
    if len(names) == 1:
        print("หมายเหตุ: เป็น dataset คลาสเดียว — Ultralytics เทรนได้ตามปกติ")


if __name__ == "__main__":
    main()
