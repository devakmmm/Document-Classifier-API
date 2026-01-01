# Document Classifier API

An ML-powered document classification system that automatically categorizes business documents (invoices, receipts, contracts, IDs) using custom logistic regression and TF-IDF features.

## Features

- 🚀 FastAPI backend with REST endpoints
- 🤖 Custom One-vs-Rest logistic regression (92% accuracy)
- 📊 TF-IDF feature extraction (20,000 n-grams)
- 🎨 Modern React UI with Tailwind CSS
- ⚡ Sub-50ms prediction latency
- 🛡️ Smart confidence thresholding with human-in-the-loop review

## Tech Stack

**Backend:** Python, FastAPI, NumPy, Scikit-learn  
**Frontend:** React, Tailwind CSS, Lucide Icons  
**ML:** TF-IDF, Logistic Regression, One-vs-Rest

## Local Development

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

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Classify document

## Deployment

Deployed on Render:
- Backend: [API URL]
- Frontend: [UI URL]

## License

MIT
