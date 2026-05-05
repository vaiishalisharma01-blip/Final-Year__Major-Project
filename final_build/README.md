# 🌾 CropSense AI — Intelligent Agriculture Assistant

A Flask-based intelligent agriculture assistant that helps farmers make data-driven decisions using Machine Learning and Deep Learning.

---

## Features

- 🌱 **Crop Recommendation** — Predicts the best crop to grow based on soil nutrients (N, P, K), pH, rainfall, and live weather data.
- 💊 **Fertilizer Recommendation** — Analyzes your soil's N/P/K levels against ideal crop requirements and advises on corrective fertilization.
- 🍃 **Plant Disease Detection** — Upload a leaf image and get an instant disease prediction using a trained ResNet9 deep learning model.
- 💬 **Agriculture Chatbot** — AI-powered chatbot (via OpenRouter API) specialized in answering farming and crop-related questions.

---

## Tech Stack

| Layer      | Technology                                     |
|------------|------------------------------------------------|
| Backend    | Python 3.x, Flask                              |
| ML Models  | scikit-learn (Random Forest, Naive Bayes)      |
| Deep Learning | PyTorch, ResNet9 CNN (38 disease classes)   |
| Data       | Pandas, NumPy                                  |
| Weather    | OpenWeatherMap API                             |
| Frontend   | HTML5, CSS3, JavaScript                        |
| Chatbot    | OpenRouter API (DeepSeek R1)                   |

---

## Project Structure

```
cropsense/
│
├── app.py                  # Flask app and all route handlers
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── utils/
│   ├── model.py            # ResNet9 architecture definition
│   └── fertilizer_disc.py  # Fertilizer recommendation text
│
├── models/
│   ├── RandomForest.pkl    # Trained crop recommendation model
│   ├── DecisionTree.pkl    # Alternative crop model
│   ├── NBClassifier.pkl    # Naive Bayes crop model
│   └── plant_disease_model.pth  # ResNet9 disease detection weights
│
├── data/
│   ├── Crop_recommendation.csv  # Training dataset
│   └── fertilizer.csv           # Ideal N/P/K values per crop
│
├── notebooks/
│   └── what-crop-to-grow.ipynb  # EDA and model training notebook
│
├── template/
│   ├── index.html          # Landing page
│   ├── crop_recom.html     # Crop recommendation form
│   ├── fertilizer.html     # Fertilizer recommendation form
│   ├── disease.html        # Disease detection (image upload)
│   └── chatbot2.html       # AI chatbot interface
│
└── static/
    ├── css/                # Stylesheets
    └── assets/             # Images and JS files
```

---

## Local Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenWeatherMap API key
Get a free key at [openweathermap.org](https://openweathermap.org/api).

**Windows:**
```cmd
set OPENWEATHER_API_KEY=your_key_here
```
**Mac/Linux:**
```bash
export OPENWEATHER_API_KEY=your_key_here
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

---

## How Each Feature Works

### Crop Recommendation
1. User enters soil N, P, K values, pH, rainfall, and city.
2. Live temperature and humidity are fetched from OpenWeatherMap.
3. All 7 features are passed to the **Random Forest** model.
4. The most suitable crop is returned.

### Fertilizer Recommendation
1. User enters current N, P, K levels and the target crop.
2. The app looks up ideal N/P/K for that crop from `fertilizer.csv`.
3. The most deficient or excess nutrient is identified.
4. Detailed remediation advice is returned.

### Plant Disease Detection
1. User uploads a leaf photo.
2. Image is resized and converted to a tensor.
3. **ResNet9** model predicts the disease class (38 possible outcomes).
4. The result is shown on screen.

---

## Supported Crops (Disease Detection)
Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — with both healthy and diseased variants (38 classes total).

---

## Disclaimer
This application is for educational and research purposes only. Always consult an agricultural expert before making farming decisions.
