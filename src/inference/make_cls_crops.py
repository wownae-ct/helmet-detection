"""
2-stage Stage-2 분류기용 머리 크롭 데이터셋 생성
================================================
기존 검출 라벨(머리 박스)에서 head 크롭을 떼어 분류 데이터셋 구성.
  - no_helmet(1) → no_helmet     (+ kaggle cap, cap_labeled 의 가까운 얼굴 포함)
  - ok(2)        → ok
  - cap(kaggle)  → no_helmet     (cap=미착용 병합)
  - fallen(0)    → 제외(자세 클래스, 분류 대상 아님)

크롭은 박스에 PAD 만큼 패딩(헬멧/머리 위·아래 맥락 포함).
추론 단계의 'person 박스 상단' 크롭과 완전히 같진 않으므로, 분류기 학습 시
강한 증강(scale/translate/erasing)으로 프레이밍 차이를 흡수한다.

출력: data/cls_crops/{train,val}/{ok,no_helmet}/*.jpg
"""
import cv2, random, xml.etree.ElementTree as ET
from pathlib import Path

random.seed(42)
BASE = Path('/mnt/ml/helmet_project')
OUT  = BASE / 'data/cls_crops'
PAD  = 0.30          # 박스 패딩 비율
MIN_PX = 16          # 너무 작은 크롭 스킵
VAL_RATIO = 0.2
IMG_EXTS = ('.jpg', '.jpeg', '.png')
# 클래스 균형: ok 크롭 상한 (no_helmet 과 비슷하게)
OK_CAP = 14000

counts = {'ok': 0, 'no_helmet': 0}
buffer = {'ok': [], 'no_helmet': []}   # (img_path, box_xywhn) 후보

def find_img(d, stem):
    for e in IMG_EXTS:
        p = d / (stem + e)
        if p.exists(): return p
    return None

def add_yolo(split_img, split_lbl):
    """data_final_backup 의 한 split 에서 no_helmet/ok 박스 수집"""
    for lbl in split_lbl.glob('*.txt'):
        img = find_img(split_img, lbl.stem)
        if img is None: continue
        with open(lbl) as f:
            for line in f:
                p = line.split()
                if len(p) < 5: continue
                cls = int(p[0])
                if cls == 1: name = 'no_helmet'
                elif cls == 2: name = 'ok'
                else: continue
                buffer[name].append((img, tuple(map(float, p[1:5]))))

def add_cap_xml(img_dir, xml_dir):
    """kaggle cap (VOC XML) → no_helmet 박스 수집"""
    for xmlp in xml_dir.glob('*.xml'):
        img = find_img(img_dir, xmlp.stem)
        if img is None: continue
        try:
            root = ET.parse(xmlp).getroot(); size = root.find('size')
            w = int(size.find('width').text); h = int(size.find('height').text)
        except Exception:
            continue
        for obj in root.findall('object'):
            if (obj.find('name').text or '').strip().lower() != 'cap': continue
            bb = obj.find('bndbox')
            x1, y1 = float(bb.find('xmin').text), float(bb.find('ymin').text)
            x2, y2 = float(bb.find('xmax').text), float(bb.find('ymax').text)
            cx, cy = (x1+x2)/2/w, (y1+y2)/2/h
            bw, bh = (x2-x1)/w, (y2-y1)/h
            buffer['no_helmet'].append((img, (cx, cy, bw, bh)))

def crop_and_save(name, items):
    out_tr = OUT / 'train' / name; out_va = OUT / 'val' / name
    out_tr.mkdir(parents=True, exist_ok=True); out_va.mkdir(parents=True, exist_ok=True)
    random.shuffle(items)
    saved = 0
    cache = {}  # 이미지 1회 로드
    for idx, (imgp, (cx, cy, bw, bh)) in enumerate(items):
        im = cache.get(imgp)
        if im is None:
            im = cv2.imread(str(imgp)); cache[imgp] = im
            if len(cache) > 200: cache.clear()  # 메모리 제한
        if im is None: continue
        H, W = im.shape[:2]
        pw, ph = bw*(1+2*PAD), bh*(1+2*PAD)
        x1 = int((cx-pw/2)*W); y1 = int((cy-ph/2)*H)
        x2 = int((cx+pw/2)*W); y2 = int((cy+ph/2)*H)
        x1, y1 = max(0,x1), max(0,y1); x2, y2 = min(W,x2), min(H,y2)
        if x2-x1 < MIN_PX or y2-y1 < MIN_PX: continue
        crop = im[y1:y2, x1:x2]
        dst = (out_va if random.random() < VAL_RATIO else out_tr) / f"{name}_{idx:06d}.jpg"
        cv2.imwrite(str(dst), crop); saved += 1
    return saved

def main():
    if OUT.exists():
        import shutil; shutil.rmtree(OUT)
    # 수집
    src = BASE / 'data/data_final_backup'
    for sp in ('train', 'valid'):
        add_yolo(src/sp/'images', src/sp/'labels')
    add_cap_xml(BASE/'data/kaggle_data/cap/cap-dataset/images',
                BASE/'data/kaggle_data/cap/cap-dataset/annotations_xml')
    print(f"수집: no_helmet={len(buffer['no_helmet']):,}  ok={len(buffer['ok']):,}")
    # ok 균형(상한)
    if len(buffer['ok']) > OK_CAP:
        random.shuffle(buffer['ok']); buffer['ok'] = buffer['ok'][:OK_CAP]
        print(f"ok 상한 적용 → {len(buffer['ok']):,}")
    # 크롭 저장
    for name in ('no_helmet', 'ok'):
        n = crop_and_save(name, buffer[name])
        counts[name] = n
        print(f"  {name}: {n:,} 크롭 저장")
    print(f"\n✅ 완료: {OUT}  (train/val, 2클래스)")

if __name__ == '__main__':
    main()
