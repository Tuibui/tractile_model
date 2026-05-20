#!/usr/bin/env python3
"""แปลงผลลัพธ์จาก CiRA CORE (YOLO/Darknet export) -> โครงสร้าง dataset ของ Ultralytics

CiRA CORE export ออกมาเป็น: รูป + ไฟล์ .txt ต่อ 1 รูป (รูปแบบ YOLO: `class_id cx cy w h`
ค่า normalize 0-1) พร้อมไฟล์รายชื่อคลาส (obj.names / classes.txt)

สคริปต์นี้จะ:
  - หาคู่ image/label ใน --src
  - อ่านไฟล์รายชื่อคลาส เพื่อคงลำดับ index ให้ตรงกับตอน label
  - แบ่ง train/val
  - คัดลอกเข้า dataset/images/{train,val} และ dataset/labels/{train,val}
  - รูปที่ไม่มี .txt จะถือเป็น negative (label ว่าง) -> เก็บไว้เทรนด้วย
  - เขียน data.yaml ให้อัตโนมัติ (ชื่อคลาสตรงกับ CiRA CORE)

ตัวอย่าง:
  python scripts/import_ciracore.py --src raw_ciracore --out dataset --val 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_FILES = ("obj.names", "classes.txt", "_darknet.labels", "classes.names")


def find_class_names(src: Path):
    for name in CLASS_FILES:
        f = src / name
        if f.exists():
            names = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
            return names, f.name
    # บางที CiRA วาง classes ไว้ในโฟลเดอร์ย่อย
    for f in src.rglob("*"):
        if f.name in CLASS_FILES:
            names = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
            return names, str(f.relative_to(src))
    return None, None


def label_for(img: Path) -> Path:
    """หาไฟล์ .txt ของรูป: ลองในโฟลเดอร์เดียวกันก่อน แล้วค่อยลอง labels/"""
    same = img.with_suffix(".txt")
    if same.exists():
        return same
    sibling = img.parent.parent / "labels" / (img.stem + ".txt")
    return sibling if sibling.exists() else same  # คืน path เดิมแม้ไม่มี (= negative)


def place(files, split, out: Path, do_copy: bool):
    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_neg = 0
    for img in files:
        dst_img = img_dir / img.name
        dst_lbl = lbl_dir / (img.stem + ".txt")
        _put(img, dst_img, do_copy)
        lbl = label_for(img)
        if lbl.exists() and lbl.stat().st_size > 0:
            _put(lbl, dst_lbl, do_copy)
        else:
            dst_lbl.write_text("")  # negative: label ว่าง
            n_neg += 1
    return n_neg


def _put(src: Path, dst: Path, do_copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if do_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="โฟลเดอร์ export จาก CiRA CORE")
    ap.add_argument("--out", default="dataset", help="โฟลเดอร์ dataset ปลายทาง")
    ap.add_argument("--val", type=float, default=0.2, help="สัดส่วน validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="คัดลอกไฟล์ (ค่าเริ่มต้น = symlink)")
    ap.add_argument("--names", nargs="*", help="กำหนดชื่อคลาสเองถ้าไม่มีไฟล์ classes")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"ไม่พบโฟลเดอร์ source: {src}")

    names, names_src = find_class_names(src)
    if names is None:
        names = args.names
    if not names:
        raise SystemExit(
            "ไม่พบไฟล์รายชื่อคลาส (obj.names/classes.txt) และไม่ได้ระบุ --names\n"
            "ใส่ --names bar_tile dot_tile หรือก๊อปไฟล์ classes มาไว้ใน --src"
        )
    print(f"คลาส ({len(names)}): {names}" + (f"  [จาก {names_src}]" if names_src else ""))

    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"ไม่พบรูปใน {src}")

    random.seed(args.seed)
    random.shuffle(images)
    n_val = int(len(images) * args.val)
    val_files, train_files = images[:n_val], images[n_val:]

    if out.exists():
        shutil.rmtree(out)
    neg_tr = place(train_files, "train", out, args.copy)
    neg_va = place(val_files, "val", out, args.copy)

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
    print(f"เขียน data.yaml -> {yaml_path}")
    print("ถ้าลำดับชื่อคลาสไม่ตรงกับ CiRA CORE ให้แก้ใน data.yaml ก่อนเทรน")


if __name__ == "__main__":
    main()
