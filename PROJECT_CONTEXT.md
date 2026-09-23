# Nirikshan — Project Context (for AI coding agents)
### Team Paakzir

> This is a trimmed context file. It contains only the technical sections a coding-agent session needs: what the product is, the four clinical exposure routes it screens for, the system architecture, tech stack, model design, the safety-net escalation rule, the exposure fusion engine spec, and the non-negotiables every coding agent must follow. Business model, market sizing, regulatory roadmap, and slide-mapping content are intentionally excluded — not needed for implementation work.

---

## 1. What we're building (one paragraph)

An AI-powered screening platform that detects the skin cancers — predominantly squamous cell carcinoma (SCC) and its precancerous precursor, actinic keratosis (AK) — caused by **chronic thermal injury, burn-scar malignant transformation, arsenic-contaminated groundwater exposure, and chronic occupational UV exposure**: four real, independently documented exposure routes that collectively affect a very large population across India, and that no existing consumer skin-cancer AI product (all of which are built around Western, light-skin, mole/melanoma screening) is built to detect. The platform combines a photo-based lesion classifier with a structured exposure-history questionnaire, reports its own accuracy broken down by skin tone rather than hiding demographic performance gaps, and is designed around a revenue model that only charges when it demonstrably prevents late-stage treatment cost — aligning with, rather than fighting, the direction India's own public health financing (PM-JAY) is already moving.

## 2. The four exposure routes (clinical scope)

| Route | Mechanism | Documented Indian evidence | Population at risk |
|---|---|---|---|
| **Chronic contact-thermal injury** | Fire-pot/brazier use (kangri, sigri, angithi, bukhari) → erythema ab igne (EAI) → SCC | Kangri-cancer clinical pathway documented since 1879 | Millions, seasonally, in northern/high-altitude India |
| **Burn-scar malignant transformation (Marjolin's ulcer)** | SCC (70–96% of cases) arising in old burn scars/chronic wounds, average latency 11–35 years, higher recurrence/metastasis risk than ordinary SCC | Published retrospective study from a Kolkata tertiary cancer center specifically on Marjolin's ulcer in India; global incidence ~1–2% of all burn scars | India carries one of the world's highest burn-injury burdens — a large population carrying old scars right now |
| **Arsenicosis-induced skin cancer** | Chronic arsenic-contaminated groundwater → hyperkeratosis → Bowen's disease/SCC/BCC | West Bengal Jalangi-block study: 1,488/7,221 screened with definite arsenical skin lesions; 70M+ people in the Bengal delta drink water above safe arsenic limits; also documented in parts of Bihar, Assam, UP | Tens of millions in arsenic-endemic groundwater belts |
| **Chronic occupational UV/actinic exposure** | Outdoor manual labor (farming, construction, fishing, vending) → actinic keratosis → SCC | Well-established global dermatology pathway; India's informal outdoor workforce is enormous | Hundreds of millions |

---

## 3. System architecture (end to end)

```
Data Sources (ISIC Archive, ISIC 2020 Challenge, BCN20000, HAM10000,
PAD-UFES-20, Fitzpatrick17k, DDI, MCSI, team-collected consented photos)
        |
        v
data/prepare_dataset.py
  -- pull, dedupe, resize to 224x224, stratified 70/15/15 split
  -- ITA (Individual Typology Angle) skin-tone estimation per image
  -- generates cropped "normal skin" negative patches from wide-field
     clinical source images (for Stage 1 negatives)
        |
        v
model/stage1_lesion_presence.py   -- binary "lesion present?" detector
model/stage2_disease_classifier.py -- EfficientNet-B0, 5-class (SCC/AK/nevus/
                                       seborrheic keratosis/healthy), transfer
                                       learning, class-weighted loss
model/checkpoints/*.pt            -- trained weights + class name order,
                                       for both stages
        |
        v
backend/ (FastAPI)
  -- POST /predict          -- image + exposure-history payload -> risk
  -- POST /risk-fusion       -- combines image-model risk + exposure score
  -- GET  /fairness-report   -- accuracy by skin-tone bin (for the dashboard)
  -- GET  /risk-map          -- aggregated, anonymized district-level stats
  -- GET  /health
        |
        v
frontend/ (React + Vite)
  -- photo upload/capture (native camera)
  -- exposure-history intake form (vernacular language toggle)
  -- risk badge (image-based risk + exposure-based risk shown separately)
  -- Grad-CAM overlay on the uploaded image
  -- fairness dashboard page
  -- offline-first service worker (queues uploads when connectivity returns)
        |
        v
Render (backend) + Vercel (frontend) -- live, public URLs, same deployment
                                          pattern as the proven prototype
```

---

## 4. Full tech stack

| Layer | Technology | Why |
|---|---|---|
| Model | PyTorch + `timm` (EfficientNet-B0, two-stage) | Same proven backbone; splitting into two stages fixes the normal-skin misclassification bug (see §6) without changing the core architecture philosophy |
| Training compute | Google Colab (T4 GPU, ideally Colab Pro for session stability) + local RTX 3050 as backup/dev only | — |
| Data tooling | `isic-cli`, `pandas`, `Pillow`, `scikit-learn`, `tqdm`, a custom ITA-estimation script | Official downloader + standard tooling + one new fairness-specific script |
| Backend | FastAPI, `uvicorn` | Fast, async, auto-docs |
| Frontend | React (Vite), plain CSS, i18n string tables for vernacular toggle | Unchanged base, one new feature layer |
| Deployment | Render (backend), Vercel (frontend) | Both have working free tiers, both deploy from GitHub |
| Versioning | Git + GitHub, GitHub Releases for model checkpoints (>100MB) | Solves the checkpoint size problem |
| CI | GitHub Actions | — |

---

## 5. Model architecture — two-stage design

**Why two stages, not one:** the old single 5-class softmax forced the model to always output a disease label, and with only ~70 "healthy" training images from one homogeneous source, it learned to key off incidental cues (framing, lighting, background of that one dataset) rather than actual skin texture — a documented failure mode in dermatology AI. Splitting the decision fixes this at the architecture level, not just the data level.

- **Stage 1 — Lesion Presence Detector.** Binary classifier (lesion / no lesion). Trained on a large, heterogeneous mix of true positives (any lesion, any class) and true negatives generated from (a) cropped normal-skin patches taken from *outside* the annotated lesion region of wide-field ISIC/BCN20000/ISIC-2020 source images, (b) the MCSI healthy subset (~100), and (c) a team-run consented photo drive for real-phone-camera diversity matching actual deployment conditions.
- **Stage 2 — Disease Classifier.** Only runs on images Stage 1 flags as "lesion present." EfficientNet-B0 transfer-learning setup: all layers frozen except the last 2 blocks + classifier head, class-weighted `CrossEntropyLoss` (weights from inverse class frequency), checkpoint selection by validation macro-F1 (not raw accuracy), checkpoint stores both `model_state_dict` and `class_names` to prevent label-order bugs.
- **Classes for Stage 2:** `squamous_cell_carcinoma`, `actinic_keratosis`, `nevus`, `seborrheic_keratosis` (four disease/control classes — SCC = malignant end-state, AK = precancerous precursor, nevus + seborrheic keratosis = hard negative controls).
- **Preprocessing:** 224×224 RGB, Pillow LANCZOS resize, ImageNet mean/std normalization. **Augmentation:** horizontal flip, ±20° rotation, brightness/contrast jitter only — **no hue/saturation jitter**, because it distorts diagnostically relevant skin-tone and lesion-color information. Skin-tone balancing is done by **re-weighting/oversampling**, not by synthetically recoloring images.

---

## 6. The safety-net escalation rule (VERBATIM — load-bearing logic)

After Stage 2 produces softmax probabilities, the system checks `p_scc` and `p_ak` specifically — not just the top-1 label:
- If `p_scc` ≥ 15% (tunable constant) and the base risk isn't already "high," escalate to "high," set `safety_escalated: true`, return a human-readable explanation.
- A parallel, slightly lower threshold applies for `p_ak`, escalating "low/none" to "moderate."
- Displayed recommendation always matches the *post-escalation* risk level, never the raw top-1 guess.
- Frontend shows a distinct amber notice when `safety_escalated` is true.

This logic is unchanged from the proven prototype and remains one of the strongest, most explainable technical talking points — keep it front and center in the pitch.

---

## 7. Exposure-History Risk Fusion Engine

The single biggest genuine innovation over the original prototype: fusing image-based risk with structured exposure history, the way a real dermatologist takes a history before examining a patient.

- **Intake form** (~10–15 questions, ~90 seconds, vernacular-language toggle): duration of fire-pot/heater use, known burn-scar history + scar age, home water source/district (checked against a static embedded arsenic-risk district list — no live API needed), outdoor occupation hours/day, sun exposure, family cancer history, tobacco/smoking use.
- **Fusion logic:** Stage 2's image-based softmax output is combined with a transparent, rules-based exposure-risk score (a documented point-scoring rubric, clinically inspired by existing risk-calculator designs — not a second opaque neural network) to produce the *final* displayed risk. The UI shows both components separately: *"image-based risk: moderate" + "exposure-based risk: high, because —"* — this is both the trust/explainability story and a natural extension of the safety-net rule already built.
- **Build ownership:** a standalone, pure-logic/data task (district risk JSON + scoring rubric + i18n strings) — no model training required, fastest agent workstream, can start immediately in parallel with data pulls.

---

## 8. Non-negotiables for coding agents (VERBATIM)

This document is the shared context to hand to each coding-agent session before its task-specific prompt (prompts to be written separately). Key non-negotiables every agent must respect, carried forward from the proven prototype's discipline:
- All file outputs should be **complete, runnable files**, not partial diffs.
- Model checkpoints store both weights and class-name order — never hardcode label order downstream.
- No hue/saturation augmentation on skin images, ever — brightness/contrast jitter only.
- Safety-net escalation logic (B.4) and the two-stage split (B.3) are both mandatory architecture, not optional.
- Every risk output must show image-based and exposure-based risk **separately**, not pre-merged.
- Checkpoint every training epoch; never assume an uninterrupted session.
