"""
VQA Answer Class Definitions — Dragon Fruit Disease Advisory App.

Maps 32 answer class IDs → full-text responses grounded in advisor.py
scientific literature.  Supports English, Vietnamese, and Thai for
farmer accessibility in Southeast-Asian dragon-fruit growing regions.

Answer taxonomy (32 classes):
    0–5   : Diagnosis     (one per disease class)
    6–10  : Severity      (None / Low / Moderate / High / Very-High)
   11–16  : Treatment     (one per disease class)
   17–22  : Prevention    (one per disease class)
   23–27  : Pathogen info (one per diseased class, Healthy excluded)
   28–31  : General       (healthy, uncertain, environmental, consult)
"""

from __future__ import annotations
import json, os
from typing import Dict, List, Tuple

# ─── CLASS NAMES (must match dataset/merged_6class order) ────────────────────
DISEASE_CLASSES: List[str] = [
    "Anthracnose",
    "Brown_Stem_Spot",
    "Gray_Blight",
    "Healthy",
    "Soft_Rot",
    "Stem_Canker",
]

NUM_ANSWER_CLASSES = 32

# ─── SUPPORTED LANGUAGES ────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = ("en", "vi", "th", "hi", "te", "kn", "ta", "ml", "mr", "bn")

# ─── ENGLISH ANSWER TEXTS ───────────────────────────────────────────────────
ANSWER_TEXTS_EN: Dict[int, str] = {
    # ── Diagnosis (0–5) ──────────────────────────────────────────────────
    0:  ("This appears to be Anthracnose — reddish-brown sunken lesions on "
         "the fruit skin with possible orange-pink spore masses."),
    1:  ("This appears to be Brown Stem Spot — small circular brown spots "
         "with yellow halos on the stems."),
    2:  ("This appears to be Gray Blight — gray to silver discoloration on "
         "stems with irregularly shaped blighted areas."),
    3:  "The plant appears healthy with no visible signs of disease.",
    4:  ("This appears to be Soft Rot — water-soaked, mushy lesions with "
         "foul-smelling bacterial exudate."),
    5:  ("This appears to be Stem Canker — orange-yellow water-soaked spots "
         "and irregular brown-black necrotic lesions on stems."),

    # ── Severity (6–10) ──────────────────────────────────────────────────
    6:  "Severity: None (0/4). The plant is healthy — no risk.",
    7:  "Severity: Low (1/4). Minor cosmetic damage — monitor closely.",
    8:  ("Severity: Moderate (2/4). Treat within 1–2 weeks to prevent "
         "further spread."),
    9:  ("Severity: High (3/4). Immediate treatment is recommended to "
         "prevent significant crop loss."),
    10: ("Severity: Very High (4/4). Aggressive intervention required — "
         "this disease can kill entire plants if left untreated."),

    # ── Treatment (11–16) ────────────────────────────────────────────────
    11: ("Treatment for Anthracnose: Apply Prochloraz or carbendazim "
         "(500 ppm) every 7–10 days during wet season. Use copper-based "
         "fungicides (copper hydroxide 77%) as a protective spray. Remove "
         "and destroy infected fruit. Postharvest: hot water (52°C, 2 min)."),
    12: ("Treatment for Brown Stem Spot: Spray propiconazole (0.1%) or "
         "difenoconazole (0.05%) at 14-day intervals. Apply foliar "
         "micronutrients (Zinc, Boron). Use copper oxychloride (0.3%). "
         "Remove severely infected cladodes."),
    13: ("Treatment for Gray Blight: Apply mancozeb (0.25%) or "
         "chlorothalonil (0.2%) sprays. Prune affected branches. Apply "
         "copper-based fungicides on pruning wounds. Improve canopy light "
         "penetration."),
    14: "No treatment required — the plant is healthy. Continue routine monitoring.",
    15: ("Treatment for Soft Rot: Remove and destroy all infected tissue "
         "immediately — do not compost. Improve drainage. Apply copper "
         "hydroxide (77%) bactericide. Disinfect tools with 10% bleach."),
    16: ("Treatment for Stem Canker: Apply 5% amino-oligosaccharide "
         "solution. Remove and burn infected cladodes ≥50 m from orchard. "
         "Apply thiophanate-methyl (0.15%) after pruning. Spray Trichoderma "
         "harzianum as a complement."),

    # ── Prevention (17–22) ───────────────────────────────────────────────
    17: ("Prevention for Anthracnose: Avoid overhead irrigation — use drip. "
         "Space plants 3×3 m. Apply preventive fungicide before rainy season. "
         "Use disease-free cuttings from certified nurseries."),
    18: ("Prevention for Brown Stem Spot: Remove fallen debris promptly. "
         "Apply organic mulch. Avoid excess nitrogen. Ensure adequate spacing."),
    19: ("Prevention for Gray Blight: Avoid mechanical injuries during "
         "harvest. Maintain balanced nutrition. Remove crop residues end of "
         "season. Monitor during high-humidity periods."),
    20: ("Prevention (Healthy): Maintain proper drainage. Apply balanced "
         "fertilization (N:P:K 10:10:10). Prune overcrowded branches. "
         "Inspect new planting material before introduction."),
    21: ("Prevention for Soft Rot: Ensure excellent field drainage using "
         "raised beds. Avoid injuries during cultivation. Reduce plant "
         "density. Water in morning — never evening."),
    22: ("Prevention for Stem Canker: Avoid wounds during cultivation. "
         "Ensure good drainage. Apply lime Ca(OH)₂ on cut surfaces. "
         "Quarantine new planting material for 2 weeks."),

    # ── Pathogen (23–27) ─────────────────────────────────────────────────
    23: ("Anthracnose is caused by Colletotrichum gloeosporioides / "
         "C. acutatum — a fungal pathogen that spreads rapidly in warm, "
         "humid conditions and can cause up to 60% postharvest loss."),
    24: ("Brown Stem Spot is caused by Bipolaris cactivora / Dothiorella "
         "spp. — a fungal pathogen favored by cool nights with high dew."),
    25: ("Gray Blight is caused by Pestalotiopsis clavispora — a secondary "
         "fungal pathogen that exploits wounds or weakened tissue."),
    26: ("Soft Rot is caused by Erwinia chrysanthemi / Pectobacterium "
         "carotovorum — a bacterial pathogen that destroys tissue via "
         "pectolytic enzymes."),
    27: ("Stem Canker is caused by Neoscytalidium dimidiatum — the most "
         "destructive fungal pathogen of dragon fruit, attacking stems "
         "and young shoots."),

    # ── General (28–31) ──────────────────────────────────────────────────
    28: "The plant looks healthy — no disease detected. Keep monitoring regularly.",
    29: ("I cannot determine the condition with high confidence. Please "
         "take a clearer photo or consult a local plant pathologist."),
    30: ("Environmental note: Dragon fruit grows optimally at 20–35°C with "
         "60–80% humidity in full sun. Disease pressure increases during "
         "monsoon season."),
    31: ("For complex or persistent issues, please consult a local "
         "agronomist or agricultural extension officer for hands-on diagnosis."),
}

# ─── VIETNAMESE ANSWER TEXTS ────────────────────────────────────────────────
ANSWER_TEXTS_VI: Dict[int, str] = {
    0:  ("Đây có vẻ là bệnh Thán thư — các tổn thương lõm màu nâu đỏ trên "
         "vỏ quả với khối bào tử màu cam-hồng."),
    1:  ("Đây có vẻ là Đốm nâu thân — các đốm tròn nhỏ màu nâu với quầng "
         "vàng trên thân."),
    2:  ("Đây có vẻ là bệnh Cháy xám — vùng đổi màu xám bạc trên thân "
         "với các vùng cháy bất thường."),
    3:  "Cây có vẻ khỏe mạnh, không có dấu hiệu bệnh.",
    4:  ("Đây có vẻ là Thối mềm — tổn thương ngấm nước, nhũn với dịch vi "
         "khuẩn có mùi hôi."),
    5:  ("Đây có vẻ là Loét thân — đốm vàng cam ngấm nước và tổn thương "
         "hoại tử nâu-đen bất thường trên thân."),
    6:  "Mức độ: Không (0/4). Cây khỏe mạnh — không có rủi ro.",
    7:  "Mức độ: Thấp (1/4). Hư hại nhẹ — theo dõi chặt chẽ.",
    8:  "Mức độ: Trung bình (2/4). Điều trị trong 1–2 tuần để ngăn lan rộng.",
    9:  "Mức độ: Cao (3/4). Cần điều trị ngay để tránh mất mùa.",
    10: "Mức độ: Rất cao (4/4). Cần can thiệp mạnh — bệnh có thể giết cây.",
    11: ("Điều trị Thán thư: Phun Prochloraz hoặc carbendazim (500 ppm) "
         "7–10 ngày/lần. Dùng thuốc gốc đồng (copper hydroxide 77%). "
         "Loại bỏ quả bệnh. Sau thu hoạch: ngâm nước nóng 52°C, 2 phút."),
    12: ("Điều trị Đốm nâu thân: Phun propiconazole (0.1%) hoặc "
         "difenoconazole (0.05%) mỗi 14 ngày. Bổ sung vi lượng Kẽm, Bo."),
    13: ("Điều trị Cháy xám: Phun mancozeb (0.25%) hoặc chlorothalonil "
         "(0.2%). Cắt tỉa cành bệnh. Bôi thuốc gốc đồng lên vết cắt."),
    14: "Không cần điều trị — cây khỏe mạnh. Tiếp tục theo dõi định kỳ.",
    15: ("Điều trị Thối mềm: Loại bỏ mô bệnh ngay — không ủ phân. Cải "
         "thiện thoát nước. Phun copper hydroxide (77%). Khử trùng dụng cụ."),
    16: ("Điều trị Loét thân: Bôi dung dịch amino-oligosaccharide 5%. Đốt "
         "cành bệnh cách vườn ≥50m. Phun thiophanate-methyl (0.15%)."),
    17: ("Phòng ngừa Thán thư: Tưới nhỏ giọt. Khoảng cách 3×3m. Phun "
         "thuốc phòng trước mùa mưa. Dùng giống sạch bệnh."),
    18: ("Phòng ngừa Đốm nâu: Dọn tàn dư. Phủ hữu cơ. Tránh thừa đạm. "
         "Đảm bảo khoảng cách thoáng."),
    19: ("Phòng ngừa Cháy xám: Tránh tổn thương cơ học khi thu hoạch. "
         "Dinh dưỡng cân đối. Dọn tàn dư cuối vụ."),
    20: ("Phòng ngừa (Khỏe): Thoát nước tốt. Bón phân cân đối NPK "
         "10:10:10. Tỉa cành rậm. Kiểm tra giống mới."),
    21: ("Phòng ngừa Thối mềm: Thoát nước tốt, luống cao. Tránh tổn "
         "thương. Giảm mật độ. Tưới sáng sớm."),
    22: ("Phòng ngừa Loét thân: Tránh tổn thương. Thoát nước tốt. Bôi "
         "vôi Ca(OH)₂ lên vết cắt. Cách ly giống mới 2 tuần."),
    23: ("Thán thư do nấm Colletotrichum gloeosporioides / C. acutatum — "
         "lây lan nhanh trong điều kiện ấm ẩm, thiệt hại đến 60%."),
    24: ("Đốm nâu thân do nấm Bipolaris cactivora / Dothiorella spp. — "
         "phát triển mạnh khi đêm mát, sương nhiều."),
    25: ("Cháy xám do nấm Pestalotiopsis clavispora — nấm thứ cấp tấn "
         "công qua vết thương hoặc mô yếu."),
    26: ("Thối mềm do vi khuẩn Erwinia chrysanthemi / Pectobacterium "
         "carotovorum — phá hủy mô bằng enzym phân giải pectin."),
    27: ("Loét thân do nấm Neoscytalidium dimidiatum — tác nhân phá hủy "
         "mạnh nhất trên thanh long, tấn công thân và chồi non."),
    28: "Cây khỏe mạnh — không phát hiện bệnh. Tiếp tục theo dõi.",
    29: ("Không thể xác định chính xác. Vui lòng chụp ảnh rõ hơn hoặc "
         "tham khảo chuyên gia bệnh cây."),
    30: ("Thanh long phát triển tối ưu ở 20–35°C, độ ẩm 60–80%, ánh sáng "
         "đầy đủ. Áp lực bệnh tăng trong mùa mưa."),
    31: ("Với vấn đề phức tạp, hãy tham khảo kỹ sư nông nghiệp hoặc "
         "trạm khuyến nông địa phương."),
}

# ─── THAI ANSWER TEXTS ──────────────────────────────────────────────────────
ANSWER_TEXTS_TH: Dict[int, str] = {
    0:  ("ดูเหมือนจะเป็นโรคแอนแทรคโนส — แผลบุ๋มสีน้ำตาลแดงบนผิวผล "
         "อาจมีกลุ่มสปอร์สีส้มชมพู"),
    1:  ("ดูเหมือนจะเป็นโรคจุดน้ำตาลที่ลำต้น — จุดกลมเล็กสีน้ำตาล "
         "มีวงแหวนสีเหลืองบนลำต้น"),
    2:  ("ดูเหมือนจะเป็นโรคใบไหม้สีเทา — บริเวณเปลี่ยนสีเป็นเทาเงิน "
         "บนลำต้น"),
    3:  "ต้นไม้ดูแข็งแรงดี ไม่มีสัญญาณของโรค",
    4:  ("ดูเหมือนจะเป็นโรคเน่าเละ — แผลฉ่ำน้ำ เละ "
         "มีของเหลวจากแบคทีเรียที่มีกลิ่นเหม็น"),
    5:  ("ดูเหมือนจะเป็นโรคแคงเกอร์ที่ลำต้น — จุดฉ่ำน้ำสีเหลืองส้ม "
         "และแผลเน่าตายสีน้ำตาลดำบนลำต้น"),
    6:  "ความรุนแรง: ไม่มี (0/4) ต้นไม้แข็งแรง ไม่มีความเสี่ยง",
    7:  "ความรุนแรง: ต่ำ (1/4) ความเสียหายเล็กน้อย ต้องเฝ้าระวัง",
    8:  "ความรุนแรง: ปานกลาง (2/4) รักษาภายใน 1–2 สัปดาห์",
    9:  "ความรุนแรง: สูง (3/4) ต้องรักษาทันทีเพื่อป้องกันการสูญเสียผลผลิต",
    10: "ความรุนแรง: สูงมาก (4/4) ต้องดำเนินการเร่งด่วน อาจทำให้ต้นตาย",
    11: ("รักษาโรคแอนแทรคโนส: ฉีดพ่น Prochloraz หรือ carbendazim (500 ppm) "
         "ทุก 7–10 วัน ใช้สารกำจัดเชื้อราทองแดง"),
    12: ("รักษาโรคจุดน้ำตาล: ฉีดพ่น propiconazole (0.1%) ทุก 14 วัน "
         "เสริมธาตุสังกะสีและโบรอน"),
    13: ("รักษาโรคใบไหม้สีเทา: ฉีดพ่น mancozeb (0.25%) หรือ "
         "chlorothalonil (0.2%) ตัดกิ่งที่ติดเชื้อ"),
    14: "ไม่ต้องรักษา — ต้นไม้แข็งแรง ดูแลตามปกติ",
    15: ("รักษาโรคเน่าเละ: เอาเนื้อเยื่อที่ติดเชื้อออกทันที "
         "ปรับปรุงการระบายน้ำ ฉีดพ่นสารทองแดง"),
    16: ("รักษาโรคแคงเกอร์: ใช้สารละลาย amino-oligosaccharide 5% "
         "เผากิ่งที่ติดเชื้อห่างสวน ≥50 เมตร"),
    17: ("ป้องกันโรคแอนแทรคโนส: ใช้ระบบน้ำหยด ระยะห่าง 3×3 เมตร "
         "ฉีดพ่นสารป้องกันก่อนฤดูฝน"),
    18: ("ป้องกันจุดน้ำตาล: เก็บเศษพืช ใช้วัสดุคลุมดิน "
         "หลีกเลี่ยงปุ๋ยไนโตรเจนมากเกินไป"),
    19: ("ป้องกันใบไหม้สีเทา: หลีกเลี่ยงบาดแผลขณะเก็บเกี่ยว "
         "ให้สารอาหารสมดุล เก็บเศษพืชหลังเก็บเกี่ยว"),
    20: ("ป้องกัน (แข็งแรง): ระบายน้ำดี ใส่ปุ๋ย NPK 10:10:10 "
         "ตัดแต่งกิ่งหนาแน่น ตรวจสอบต้นพันธุ์ใหม่"),
    21: ("ป้องกันเน่าเละ: ระบายน้ำดีด้วยแปลงยกร่อง หลีกเลี่ยงบาดแผล "
         "รดน้ำตอนเช้า"),
    22: ("ป้องกันแคงเกอร์: หลีกเลี่ยงบาดแผล ระบายน้ำดี "
         "ทาปูนขาวบนรอยตัด กักกันต้นพันธุ์ใหม่ 2 สัปดาห์"),
    23: ("โรคแอนแทรคโนสเกิดจากเชื้อรา Colletotrichum gloeosporioides "
         "แพร่กระจายเร็วในสภาพอากาศอุ่นชื้น"),
    24: ("จุดน้ำตาลเกิดจากเชื้อรา Bipolaris cactivora "
         "เจริญเติบโตดีในคืนที่อากาศเย็นและมีน้ำค้าง"),
    25: ("ใบไหม้สีเทาเกิดจากเชื้อรา Pestalotiopsis clavispora "
         "เชื้อราทุติยภูมิที่เข้าทำลายผ่านบาดแผล"),
    26: ("เน่าเละเกิดจากแบคทีเรีย Erwinia chrysanthemi / "
         "Pectobacterium carotovorum ทำลายเนื้อเยื่อด้วยเอนไซม์"),
    27: ("แคงเกอร์เกิดจากเชื้อรา Neoscytalidium dimidiatum "
         "เชื้อราที่ทำลายรุนแรงที่สุดในแก้วมังกร"),
    28: "ต้นไม้แข็งแรง ไม่พบโรค ดูแลตามปกติ",
    29: ("ไม่สามารถระบุได้แน่ชัด กรุณาถ่ายภาพใหม่ให้ชัดขึ้น "
         "หรือปรึกษานักโรคพืช"),
    30: ("แก้วมังกรเจริญเติบโตดีที่ 20–35°C ความชื้น 60–80% "
         "โรคมักระบาดในฤดูฝน"),
    31: ("สำหรับปัญหาที่ซับซ้อน กรุณาปรึกษานักวิชาการเกษตร "
         "หรือเจ้าหน้าที่ส่งเสริมการเกษตรท้องถิ่น"),
}

# ─── LANGUAGE REGISTRY ───────────────────────────────────────────────────────
ANSWER_TEXTS: Dict[str, Dict[int, str]] = {
    "en": ANSWER_TEXTS_EN,
    "vi": ANSWER_TEXTS_VI,
    "th": ANSWER_TEXTS_TH,
}

# Auto-load Indic translations from mobile_answer_dict.json if available
_dict_path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "mobile_answer_dict.json")
if os.path.exists(_dict_path):
    try:
        with open(_dict_path, "r", encoding="utf-8") as _f:
            _all = json.load(_f)
        for _lang in ("hi", "te", "kn", "ta", "ml", "mr", "bn"):
            if _lang in _all:
                ANSWER_TEXTS[_lang] = {int(k): v for k, v in _all[_lang].items()}
    except Exception:
        pass

# ─── DISEASE → ANSWER ID MAPPING ────────────────────────────────────────────
# Maps (disease_class, question_type) → answer_class_id
# This is the core lookup used by both the dataset generator and inference.

_DISEASE_TO_IDX = {d: i for i, d in enumerate(DISEASE_CLASSES)}

# Diseases that have pathogen entries (all except Healthy)
_DISEASED_CLASSES = [d for d in DISEASE_CLASSES if d != "Healthy"]
_DISEASED_TO_PATHOGEN_ID = {
    "Anthracnose":     23,
    "Brown_Stem_Spot": 24,
    "Gray_Blight":     25,
    "Soft_Rot":        26,
    "Stem_Canker":     27,
}

# Severity score → severity answer ID
_SEVERITY_SCORES = {
    "Healthy":         0,
    "Anthracnose":     3,
    "Brown_Stem_Spot": 2,
    "Gray_Blight":     2,
    "Soft_Rot":        3,
    "Stem_Canker":     4,
}

_SEVERITY_TO_ANSWER = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10}


def get_answer_id(disease_class: str, question_type: str) -> int:
    """
    Map (disease_class, question_type) → answer_class_id.

    Args:
        disease_class: One of DISEASE_CLASSES
        question_type: One of 'diagnosis', 'severity', 'treatment',
                       'prevention', 'pathogen', 'general'

    Returns:
        Integer answer class ID in [0, 31]
    """
    idx = _DISEASE_TO_IDX.get(disease_class, 3)   # default to Healthy

    if question_type == "diagnosis":
        return idx                                  # 0–5

    if question_type == "severity":
        score = _SEVERITY_SCORES.get(disease_class, 0)
        return _SEVERITY_TO_ANSWER[score]           # 6–10

    if question_type == "treatment":
        return 11 + idx                             # 11–16

    if question_type == "prevention":
        return 17 + idx                             # 17–22

    if question_type == "pathogen":
        if disease_class in _DISEASED_TO_PATHOGEN_ID:
            return _DISEASED_TO_PATHOGEN_ID[disease_class]  # 23–27
        return 28                                   # healthy → "plant looks healthy"

    # General fallback
    if disease_class == "Healthy":
        return 28
    return 31                                       # consult agronomist


def get_answer_text(answer_id: int, lang: str = "en") -> str:
    """Return the full-text response for a given answer class and language."""
    texts = ANSWER_TEXTS.get(lang, ANSWER_TEXTS_EN)
    return texts.get(answer_id, ANSWER_TEXTS_EN.get(answer_id, ""))


def export_answer_map(path: str) -> None:
    """Save all answer texts (all languages) to a JSON file for edge deployment."""
    payload = {
        lang: {str(k): v for k, v in texts.items()}
        for lang, texts in ANSWER_TEXTS.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ─── QUICK SANITY CHECK ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Answer classes: {NUM_ANSWER_CLASSES}")
    print(f"Languages: {SUPPORTED_LANGUAGES}")
    for lang in SUPPORTED_LANGUAGES:
        texts = ANSWER_TEXTS[lang]
        print(f"  [{lang}] {len(texts)} entries — sample[0]: {texts[0][:60]}…")

    # Test mapping
    for disease in DISEASE_CLASSES:
        for qtype in ("diagnosis", "severity", "treatment", "prevention", "pathogen"):
            aid = get_answer_id(disease, qtype)
            print(f"  {disease:20s} × {qtype:12s} → answer_id={aid:2d}")
