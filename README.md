# Nirikshan
### Team Paakzir

An AI-powered screening platform that detects the skin cancers — predominantly squamous cell carcinoma (SCC) and its precancerous precursor, actinic keratosis (AK) — caused by **chronic thermal injury, burn-scar malignant transformation, arsenic-contaminated groundwater exposure, and chronic occupational UV exposure**: four real, independently documented exposure routes that collectively affect a very large population across India, and that no existing consumer skin-cancer AI product (all of which are built around Western, light-skin, mole/melanoma screening) is built to detect. The platform combines a photo-based lesion classifier with a structured exposure-history questionnaire, reports its own accuracy broken down by skin tone rather than hiding demographic performance gaps, and is designed around a revenue model that only charges when it demonstrably prevents late-stage treatment cost — aligning with, rather than fighting, the direction India's own public health financing (PM-JAY) is already moving.

## Architecture

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

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Model | PyTorch + `timm` (EfficientNet-B0, two-stage) | Same proven backbone; splitting into two stages fixes the normal-skin misclassification bug without changing the core architecture philosophy |
| Training compute | Google Colab (T4 GPU, ideally Colab Pro for session stability) + local RTX 3050 as backup/dev only | — |
| Data tooling | `isic-cli`, `pandas`, `Pillow`, `scikit-learn`, `tqdm`, a custom ITA-estimation script | Official downloader + standard tooling + one new fairness-specific script |
| Backend | FastAPI, `uvicorn` | Fast, async, auto-docs |
| Frontend | React (Vite), plain CSS, i18n string tables for vernacular toggle | Unchanged base, one new feature layer |
| Deployment | Render (backend), Vercel (frontend) | Both have working free tiers, both deploy from GitHub |
| Versioning | Git + GitHub, GitHub Releases for model checkpoints (>100MB) | Solves the checkpoint size problem |
| CI | GitHub Actions | — |

## Status

- [ ] Data pipeline
- [ ] Stage-1 model (lesion presence detector)
- [ ] Stage-2 model (disease classifier)
- [ ] Exposure fusion engine
- [ ] Backend (FastAPI)
- [ ] Frontend (React + Vite)
- [ ] Deployment (Render + Vercel)

## Cost

The whole stack runs on free tiers only — no credit card required anywhere.
