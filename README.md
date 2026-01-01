# Document Classifier API

An ML-powered document classification system that automatically categorizes business documents (invoices, receipts, contracts, IDs) using custom logistic regression and TF-IDF features.

## 🚀 Live Demo

- **Frontend:** https://document-classifier-ui.onrender.com
- **Backend API:** https://document-classifier-api-fewk.onrender.com
- **API Docs:** https://document-classifier-api-fewk.onrender.com/docs

## ✨ Features

- 🤖 Custom One-vs-Rest logistic regression (92% accuracy)
- ⚡ Sub-50ms prediction latency
- 📊 TF-IDF feature extraction (20,000 n-grams)
- 🎨 Modern React UI with Tailwind CSS
- 🛡️ Smart confidence thresholding with human-in-the-loop review
- 🔄 Real-time classification via REST API

## 🛠️ Tech Stack

**Backend:** Python, FastAPI, NumPy, Scikit-learn  
**Frontend:** React, Tailwind CSS, Lucide Icons  
**ML:** TF-IDF, Logistic Regression, One-vs-Rest  
**Deployment:** Render (Web Service + Static Site)

## 📦 Local Development

### Backend
```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset
python -m src.make_dataset

# Train model
python -m src.train

# Start API server
python -m uvicorn src.api:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## 📡 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /ping` - Keep-alive ping
- `POST /predict` - Classify document
- `GET /docs` - Interactive API documentation

## 📝 Example Request
```bash
curl -X POST https://document-classifier-api-fewk.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Invoice #8472 Total Due $1250.50. Payment due within 30 days.",
    "top_k": 3,
    "threshold": 0.65
  }'
```

## 🎯 Classification Categories

- **Invoice** - Bills requesting payment
- **Receipt** - Proof of purchase
- **Contract** - Legal agreements
- **ID** - Identification documents

## 🔧 Configuration

Update `frontend/src/App.js` to point to your backend:
```javascript
const response = await fetch('YOUR_BACKEND_URL/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text, top_k: 3, threshold: 0.65 })
});
```

## 📈 Performance Metrics

- **Accuracy:** ~92% on test data
- **Inference Time:** <50ms per document
- **Training Time:** <2 minutes on 540 samples
- **Features:** 20,000 TF-IDF n-grams (unigrams + bigrams)

## 📄 License

MIT

## 👤 Author

Devak Mehta - [LinkedIn](https://www.linkedin.com/in/devak-mehta) | [GitHub](https://github.com/devakmmm)