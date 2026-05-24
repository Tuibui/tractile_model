#!/usr/bin/env bash
# โหมดพับจอเดินเก็บ: อัดทันที (--autostart) + ไม่มีพรีวิว (--no-preview)
# ใช้งาน:
#   ./record.sh                         # อัดลง recordings/ ทันที, หยุดด้วย Ctrl+C
#   ./record.sh --orientation rot180    # กล้องกลับหัว
#   ./record.sh --out myclips           # เปลี่ยนโฟลเดอร์ปลายทาง
set -e
cd "$(dirname "$0")"
VENV_PY=/home/tuibui/collect_image/venv/bin/python

# พับจอแล้วเครื่องอย่าหลับ "เฉพาะตอนอัด" ไม่งั้น USB โดนตัดไฟ -> OAK-D-Lite หลุด (X_LINK_ERROR)
# ต้องตั้ง LidSwitchIgnoreInhibited=no ใน logind ด้วย ไม่งั้น inhibitor นี้จะถูกเมิน
exec systemd-inhibit \
  --what=handle-lid-switch:sleep:idle \
  --who="record.sh" \
  --why="OAK-D-Lite recording (fold-screen walk mode)" \
  --mode=block \
  "$VENV_PY" scripts/record_oak.py --no-preview --autostart --out recordings "$@"
