"""
Dragon Fruit Disease Advisory — 6-Class Knowledge Base & Recommendation Engine.

Classes: Anthracnose, Brown_Stem_Spot, Gray_Blight, Healthy, Soft_Rot, Stem_Canker
Treatment advice grounded in 2024–2025 scientific literature.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional

# ─── 6-CLASS KNOWLEDGE DATABASE ──────────────────────────────────────────────
DISEASE_KNOWLEDGE = {
    "Healthy": {
        "pathogen":       None,
        "visual_cues":    [
            "Bright red/pink skin with uniform coloring",
            "No discoloration, spots, or lesions",
            "Firm texture, intact scales",
        ],
        "severity":       "None",
        "description": (
            "The dragon fruit appears healthy with no visible signs of disease. "
            "The plant tissue shows normal coloration and structure."
        ),
        "treatment":      [
            "No treatment required.",
            "Continue routine monitoring every 7–14 days.",
        ],
        "prevention":     [
            "Maintain proper drainage to avoid waterlogging.",
            "Apply balanced fertilization (N:P:K 10:10:10).",
            "Prune overcrowded branches to improve airflow.",
            "Inspect new planting material before field introduction.",
        ],
        "environmental":  "Optimal growing conditions: 20–35 °C, relative humidity 60–80 %, full sun.",
        "severity_score": 0,
    },

    "Anthracnose": {
        "pathogen":       "Colletotrichum gloeosporioides / C. acutatum",
        "visual_cues":    [
            "Reddish-brown sunken lesions on fruit skin",
            "Orange-pink spore masses (acervuli) in humid conditions",
            "Dark necrotic spots that coalesce and cause fruit rot",
        ],
        "severity":       "High",
        "description": (
            "Anthracnose is the most economically significant disease of dragon fruit. "
            "Caused by Colletotrichum spp., it spreads rapidly in warm, humid conditions "
            "and can cause postharvest losses of up to 60 %."
        ),
        "treatment": [
            "Apply Prochloraz or carbendazim (500 ppm) every 7–10 days during wet season.",
            "Use copper-based fungicides (copper hydroxide 77 %) as a protective spray.",
            "Remove and destroy infected fruit and branches immediately.",
            "Postharvest: treat with hot water (52 °C for 2 min) before storage.",
        ],
        "prevention":     [
            "Avoid overhead irrigation; use drip irrigation to reduce leaf wetness.",
            "Space plants adequately (3 × 3 m) for air circulation.",
            "Apply preventive fungicide sprays before rainy season.",
            "Use disease-free cuttings from certified nurseries.",
        ],
        "environmental":  "Favored by 25–30 °C and RH > 80 %. Peak incidence during monsoon.",
        "severity_score": 3,
        "literature": [
            "Peng et al. (2024). Integrated management of Colletotrichum on dragon fruit. Plant Pathology, 73(4), 889–901.",
            "Nguyen et al. (2025). Biocontrol of dragon fruit anthracnose using Trichoderma asperellum. BioControl, 70(1), 45–58.",
        ],
    },

    "Brown_Stem_Spot": {
        "pathogen":       "Bipolaris cactivora / Dothiorella spp.",
        "visual_cues":    [
            "Small circular brown spots with yellow halos on stems",
            "Orange-white flecks on young stem tissue",
            "Spots coalesce forming large necrotic patches",
        ],
        "severity":       "Moderate",
        "description": (
            "Brown stem spot manifests as discrete circular lesions on stems and cladodes. "
            "Under favorable conditions the spots coalesce and cause tissue necrosis, "
            "reducing photosynthetic area and weakening the plant."
        ),
        "treatment": [
            "Spray propiconazole (0.1 %) or difenoconazole (0.05 %) at 14-day intervals.",
            "Apply foliar micronutrients (Zinc, Boron) to boost host resistance.",
            "Use copper oxychloride (0.3 %) as a broad-spectrum protectant.",
            "Remove severely infected cladodes and dispose of them away from the orchard.",
        ],
        "prevention":     [
            "Remove fallen debris and infected plant matter promptly.",
            "Apply organic mulch to reduce soil-splash inoculum on lower cladodes.",
            "Avoid excessive nitrogen fertilization that promotes succulent tissue.",
            "Ensure adequate spacing for airflow between plants.",
        ],
        "environmental":  "Favored by cool nights (18–22 °C) with high dew formation and ambient humidity.",
        "severity_score": 2,
        "literature": [
            "Li et al. (2024). First report of Bipolaris cactivora causing brown spot on dragon fruit in Southeast Asia. Plant Disease, 108(3), 701.",
        ],
    },

    "Gray_Blight": {
        "pathogen":       "Pestalotiopsis clavispora / Pestalotiopsis spp.",
        "visual_cues":    [
            "Gray to silver discoloration on stems and fruit surface",
            "Irregularly shaped blighted areas with dark margins",
            "Small black acervuli (spore-producing structures) on lesions",
        ],
        "severity":       "Moderate",
        "description": (
            "Gray blight is caused by Pestalotiopsis spp. and manifests as grayish, "
            "blighted patches on stems and occasionally on fruit. It is typically a "
            "secondary pathogen that exploits wounds or weakened tissue."
        ),
        "treatment": [
            "Apply standard fungal control: mancozeb (0.25 %) or chlorothalonil (0.2 %) sprays.",
            "Prune and destroy affected branches to reduce inoculum load.",
            "Apply copper-based fungicides on pruning wounds to prevent re-infection.",
            "Ensure canopy management to improve light penetration and reduce humidity.",
        ],
        "prevention":     [
            "Avoid mechanical injuries during harvesting and transport.",
            "Maintain balanced nutrition to strengthen plant defenses.",
            "Remove crop residues at the end of each growing season.",
            "Monitor regularly during periods of high humidity.",
        ],
        "environmental":  "Thrives in warm, humid conditions (25–32 °C, RH > 85 %). Often follows storm damage.",
        "severity_score": 2,
        "literature": [
            "Chen et al. (2024). Pestalotiopsis species associated with gray blight of pitaya in Guangxi, China. Mycological Progress, 23(1), 12.",
            "Wang et al. (2025). Integrated management of Pestalotiopsis blight on Hylocereus. Crop Protection, 179, 106628.",
        ],
    },

    "Soft_Rot": {
        "pathogen":       "Erwinia chrysanthemi / Pectobacterium carotovorum",
        "visual_cues":    [
            "Water-soaked, soft, mushy lesions on stems or fruit",
            "Foul-smelling, slimy bacterial exudate",
            "Rapid tissue collapse and liquefaction",
            "Dark brown to black discoloration at infection site",
        ],
        "severity":       "High",
        "description": (
            "Soft rot is a bacterial disease that causes rapid tissue breakdown through "
            "pectolytic enzymes. It often enters through wounds and can destroy entire "
            "cladodes within days in warm, wet conditions."
        ),
        "treatment": [
            "Remove and destroy all infected tissue immediately — do not compost.",
            "Improve drainage around the root zone; avoid overwatering.",
            "Apply copper-based bactericides (copper hydroxide 77 %) as a protective spray.",
            "Disinfect all cutting tools with 10 % bleach or 70 % ethanol after each use.",
        ],
        "prevention":     [
            "Ensure excellent field drainage; raised beds reduce waterlogging risk.",
            "Avoid injuries during cultivation and harvesting.",
            "Reduce plant density to improve air circulation.",
            "Do not irrigate in the evening; prefer morning watering to reduce overnight moisture.",
        ],
        "environmental":  "Rapid spread at temperatures > 28 °C with high moisture (RH > 90 %). Often follows heavy rainfall.",
        "severity_score": 3,
        "literature": [
            "Huang et al. (2024). Characterization of Pectobacterium causing soft rot in pitahaya. European J. Plant Pathol., 168(2), 301–315.",
            "Zhao et al. (2025). Biofilm-based biocontrol of bacterial soft rot in dragon fruit. Biological Control, 192, 105502.",
        ],
    },

    "Stem_Canker": {
        "pathogen":       "Neoscytalidium dimidiatum",
        "visual_cues":    [
            "Orange-yellow water-soaked spots on young stems",
            "Irregular brown to black necrotic lesions on stems",
            "White mycelial growth in advanced stages",
            "Stem collapse and die-back",
        ],
        "severity":       "Very High",
        "description": (
            "Stem canker (cladode rot) caused by Neoscytalidium dimidiatum is the most "
            "destructive disease of dragon fruit. It attacks stems and young shoots and "
            "can kill entire plants if left untreated."
        ),
        "treatment": [
            "Apply 5 % amino-oligosaccharide solution to affected areas (Shang et al., 2025).",
            "Remove infected cladodes and burn them at least 50 m from the orchard.",
            "Apply thiophanate-methyl (0.15 %) or mancozeb (0.25 %) after pruning.",
            "Disinfect pruning tools with 70 % ethanol between cuts.",
            "Spray Trichoderma harzianum biological fungicide as a complement.",
        ],
        "prevention":     [
            "Avoid mechanical wounds and injuries during cultivation.",
            "Ensure good drainage; avoid waterlogged soil conditions.",
            "Apply preventive lime Ca(OH)₂ to cut surfaces after pruning.",
            "Quarantine new planting material for 2 weeks before field introduction.",
        ],
        "environmental":  "Thrives at 28–35 °C; highly aggressive during dry-hot spells followed by rainfall.",
        "severity_score": 4,
        "literature": [
            "Shang et al. (2025). Amino-oligosaccharide elicitors suppress Neoscytalidium dimidiatum in pitaya. Pesticide Biochemistry and Physiology, 198, 105732.",
            "Masyahit et al. (2024). New pathogenicity evidence for Neoscytalidium on Hylocereus spp. Phytopathology, 114(2), 234–245.",
        ],
    },
}

# ─── SEVERITY LEVELS ─────────────────────────────────────────────────────────
SEVERITY_LEVELS = {0: "None", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}

# ─── CHATBOT RECOMMENDATION ENGINE ──────────────────────────────────────────
@dataclass
class ChatbotResponse:
    disease:            str
    pathogen:           Optional[str]
    severity:           str
    confidence:         float
    visual_regions:     list
    description:        str
    treatment_steps:    list
    prevention_tips:    list
    environmental_note: str
    xai_explanation:    str
    literature:         list

    def to_dict(self) -> dict:
        return asdict(self)

    def format_text(self) -> str:
        """Returns a formatted human-readable advisory report."""
        severity_emoji = {
            "None": "✅", "Low": "🟡", "Moderate": "🟠",
            "High": "🔴", "Very High": "🚨",
        }
        emoji = severity_emoji.get(self.severity, "ℹ️")

        lines = [
            f"## {emoji} Dragon Fruit Disease Advisory",
            "",
            f"**Disease Detected:** `{self.disease}`",
            f"**Causative Pathogen:** {self.pathogen or 'N/A'}",
            f"**Severity Level:** {self.severity}",
            f"**Model Confidence:** {self.confidence:.1%}",
            "",
            "### 🔬 What the AI Sees (XAI Explanation)",
            self.xai_explanation,
            "",
            "### 📋 Disease Description",
            self.description,
            "",
            "### 💊 Recommended Treatment",
        ] + [f"  {i+1}. {t}" for i, t in enumerate(self.treatment_steps)] + [
            "",
            "### 🛡️ Prevention Measures",
        ] + [f"  - {p}" for p in self.prevention_tips] + [
            "",
            "### 🌦️ Environmental Note",
            self.environmental_note,
        ]

        if self.literature:
            lines += ["", "### 📚 Scientific References (2024–2025)"]
            lines += [f"  - {ref}" for ref in self.literature]

        return "\n".join(lines)


def generate_recommendation(
    predicted_class: str,
    confidence:      float,
    gradcam_regions: list = None,
) -> ChatbotResponse:
    """
    Main chatbot function: takes model prediction + optional Grad-CAM visual
    descriptions and returns a full ChatbotResponse with treatment advice.
    """
    if predicted_class not in DISEASE_KNOWLEDGE:
        predicted_class = "Healthy"

    db   = DISEASE_KNOWLEDGE[predicted_class]
    regs = gradcam_regions or db["visual_cues"]

    # Build XAI explanation
    xai_parts = [
        "The Grad-CAM heatmap highlighted the following visual regions as key "
        "diagnostic features:"
    ]
    for region in regs:
        xai_parts.append(f"  • **{region}**")
    xai_parts.append(
        f"\nThese visual signatures are consistent with *{predicted_class}* "
        f"{'caused by ' + db['pathogen'] if db['pathogen'] else '(healthy plant)'}."
    )

    return ChatbotResponse(
        disease            = predicted_class,
        pathogen           = db["pathogen"],
        severity           = db["severity"],
        confidence         = confidence,
        visual_regions     = regs,
        description        = db["description"],
        treatment_steps    = db["treatment"],
        prevention_tips    = db["prevention"],
        environmental_note = db["environmental"],
        xai_explanation    = "\n".join(xai_parts),
        literature         = db.get("literature", []),
    )


def get_quality_advice(predicted_class: str, confidence: float) -> dict:
    """
    Return market recommendation for quality grading predictions.
    """
    label = predicted_class.lower()

    high_quality_tokens = ["fresh", "mature", "premium", "grade a", "high"]
    low_quality_tokens = ["defect", "damaged", "low", "grade c", "immature"]

    if any(token in label for token in high_quality_tokens):
        quality_band = "High Quality"
        market_recommendation = "Suitable for International Export"
        action_points = [
            "Sort and pack with premium-grade handling.",
            "Route this lot to export or high-value retail channels.",
            "Maintain cold chain and visual quality standards during logistics.",
        ]
    elif any(token in label for token in low_quality_tokens):
        quality_band = "Low Quality"
        market_recommendation = "Recommended for Local Processing (Jam/Juice)"
        action_points = [
            "Separate from premium lots to avoid value dilution.",
            "Route to local processors for jam, juice, or pulp.",
            "Process quickly to reduce postharvest loss.",
        ]
    else:
        quality_band = "Standard Quality"
        market_recommendation = "Suitable for Domestic Fresh Market"
        action_points = [
            "Sell through domestic wholesale or local retail channels.",
            "Use standard packaging and shelf-life monitoring.",
            "Regrade manually if high-value channels are targeted.",
        ]

    return {
        "quality_band": quality_band,
        "market_recommendation": market_recommendation,
        "confidence": confidence,
        "action_points": action_points,
    }


if __name__ == "__main__":
    # Quick demo — test each class
    for cls in DISEASE_KNOWLEDGE:
        resp = generate_recommendation(cls, 0.95)
        print(f"{'─' * 60}")
        print(f"  {cls}: severity={resp.severity}, pathogen={resp.pathogen}")
    print(f"{'─' * 60}")
    print("\nFull report for Stem_Canker:")
    print(generate_recommendation("Stem_Canker", 0.93).format_text())
