# Rider Intention Prediction Project

A comparative analysis of CNN-LSTM vs 3D CNN models for predicting motorcycle rider intentions using simulated traffic data.

## Project Structure
- `src/` - Source code for models and training
- `web_app/` - Flask web interface for inference
- `data/` - Sample data and data generation scripts
- `models/` - Trained model files
- `.github/workflows/` - CI/CD pipelines

## Setup
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train.py
