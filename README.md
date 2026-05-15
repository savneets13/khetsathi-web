# KhetSathi 🌱

> **Crop Disease Detection for Indian Agriculture using Deep Learning**
>
> BCA Major Project — Final Year — Amity University Online

A smartphone-accessible web app that identifies crop diseases from leaf photographs. Powered by a MobileNetV2 model trained on the PlantVillage dataset, fine-tuned with transfer learning, and deployed as a Streamlit web application.

## What This Does

Snap a photo of a leaf from tomato, potato, or bell pepper plants. The model identifies which of 13 conditions the leaf shows — including 10 disease classes and 3 healthy classes. Each result includes symptoms description, treatment recommendations, prevention practices, and severity indicator.

## Why This Matters

Indian farmers lose 15-25% of their crop yield to plant diseases every year. Most farmers don't have direct access to agricultural extension officers, but over 750 million Indians now own smartphones. Mobile-deployed deep learning can bridge this gap — putting expert-level disease identification in every farmer's pocket.

## Live Demo

🔗 **[Open KhetSathi](https://khetsathi.streamlit.app)** (works on any phone or computer browser)

## Model Performance

| Metric | Value |
|---|---|
| Architecture | MobileNetV2 (Transfer Learning from ImageNet) |
| Training Data | PlantVillage dataset (14,005 images) |
| Test Accuracy | **95.32%** on 3,015 held-out test images |
| Model Size | 5.03 MB (Core ML, FP16 quantized) |
| Inference Time | ~50ms on iPhone 12+ |

## Supported Crops & Diseases

**Bell Pepper:** Healthy, Bacterial Spot
**Potato:** Healthy, Early Blight, Late Blight
**Tomato:** Healthy, Bacterial Spot, Late Blight, Leaf Mold, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus

## Tech Stack

- **ML Framework:** TensorFlow 2.x / Keras
- **Architecture:** MobileNetV2 + custom classifier head (frozen base + fine-tuned head)
- **Web Frontend:** Streamlit
- **Mobile Frontend:** SwiftUI (iOS 17+, separate repo)
- **Mobile Inference:** Core ML
- **Cloud Hosting:** Streamlit Community Cloud (free tier)

## Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/khetsathi-web.git
cd khetsathi-web

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## File Structure

```
khetsathi-web/
├── streamlit_app.py          # Main Streamlit app
├── KhetSathiModel.tflite     # Trained model (TFLite format)
├── disease_info.json         # Disease info for all 13 classes
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Project Context

This is the web deployment component of a BCA final-year major project. The complete project includes:

1. **Web app (this repo)** — accessible from any smartphone browser
2. **Native iOS app** — KhetSathi iOS, using Core ML (separate repo, in development)
3. **Project report** — full academic documentation following APA 7 guidelines

## Author

**[Your Full Name]**
BCA, Amity University Online — Final Year
Enrolment: [Your Enrolment Number]

## Acknowledgements

- **PlantVillage Dataset** — Hughes & Salathé (2015)
- **MobileNetV2 Architecture** — Sandler et al. (2018)
- **Transfer Learning Methodology** — Building on work by Mohanty et al. (2016), Ferentinos (2018), Too et al. (2019)

## License

MIT License — see LICENSE file for details.

---

*This project is submitted in partial fulfillment of the requirements for the Bachelor of Computer Applications (BCA) degree.*
