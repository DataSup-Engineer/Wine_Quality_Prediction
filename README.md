# Wine Quality Classifier 🍷

A machine learning web application that predicts wine quality scores (0-10) based on physicochemical properties using logistic regression.

## Overview

This project implements a wine quality classification system trained on the Portuguese "Vinho Verde" wine dataset. The system analyzes 11 physicochemical properties to predict wine quality scores, providing both a user-friendly web interface and a JSON API for predictions.

**Note**: The model is trained on physicochemical properties only and does not use wine type (red/white) as a feature, making predictions based purely on measurable chemical characteristics.

## Features

- **Machine Learning Model**: Logistic regression classifier trained on 6,497 wine samples
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique) for better minority class prediction
- **Web Interface**: Clean, intuitive Flask-based web application
- **API Endpoint**: RESTful API for programmatic access
- **Comprehensive Analysis**: Exploratory data analysis notebook included
- **Property-Based Testing**: Hypothesis-based tests for robust validation

## Dataset

The project uses the Wine Quality Dataset containing:
- **Red wines**: 1,599 samples
- **White wines**: 4,898 samples
- **Features**: 11 physicochemical properties
- **Target**: Quality scores (typically 3-9)

### Physicochemical Properties

1. Fixed Acidity (3.8 - 15.9)
2. Volatile Acidity (0.08 - 1.58)
3. Citric Acid (0.0 - 1.66)
4. Residual Sugar (0.6 - 65.8)
5. Chlorides (0.009 - 0.611)
6. Free Sulfur Dioxide (1 - 289)
7. Total Sulfur Dioxide (6 - 440)
8. Density (0.987 - 1.039)
9. pH (2.72 - 4.01)
10. Sulphates (0.22 - 2.0)
11. Alcohol (8.0 - 14.9%)

## Project Structure

```
wine-quality-classifier/
├── data/                          # Data storage
│   ├── raw/                       # Original CSV files
│   └── processed/                 # Processed datasets
├── models/                        # Trained models
│   ├── trained_model.pkl          # Logistic regression model
│   └── scaler.pkl                 # Feature scaler
├── notebooks/                     # Jupyter notebooks
│   └── eda.ipynb                  # Exploratory data analysis
├── src/                           # Source code
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── model.py                   # Model training and prediction
│   ├── evaluation.py              # Model evaluation utilities
│   └── app.py                     # Flask web application
├── templates/                     # HTML templates
│   ├── index.html                 # Input form
│   └── result.html                # Prediction results
├── static/                        # Static assets
│   └── style.css                  # Stylesheet
├── tests/                         # Test suite
│   └── test_data_loader_properties.py  # Property-based tests
├── train.py                       # Training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

The required packages include:
- scikit-learn (machine learning)
- pandas (data manipulation)
- numpy (numerical computing)
- imbalanced-learn (SMOTE for class imbalance)
- matplotlib & seaborn (visualization)
- flask (web framework)
- joblib (model persistence)
- hypothesis (property-based testing)
- pytest (testing framework)

## Usage

### 1. Train the Model

Before using the web application, you need to train the model:

```bash
python train.py
```

This script will:
- Load and preprocess the wine quality datasets
- Split data into training (70%) and test (30%) sets
- Apply SMOTE to balance minority classes in training data
- Train a logistic regression model on balanced data
- Evaluate model performance on original (unbalanced) test set
- Save the trained model and scaler to `models/`

**SMOTE Class Balancing:**
The training process uses targeted oversampling to improve minority class representation:
- Quality 3: 21 → 500 samples
- Quality 4: 151 → 800 samples
- Quality 8: 135 → 500 samples
- Quality 9: 4 → 300 samples

**Expected Output:**
- Training samples: ~4,547 → ~6,336 (after SMOTE)
- Test accuracy: ~48%
- Improved recall for minority classes (3, 4, 8, 9)
- Model files: `models/trained_model.pkl` and `models/scaler.pkl`

### 2. Run the Flask Application

Start the web server:

```bash
python src/app.py
```

The application will be available at: **http://127.0.0.1:5001**

### 3. Make Predictions

#### Via Web Interface

1. Open your browser and navigate to `http://127.0.0.1:5001`
2. Enter the wine's 11 physicochemical properties in the form
3. Click "Predict Quality"
4. View the predicted quality score and confidence

#### Via API

Send a POST request to `/api/predict` with JSON data:

```bash
curl -X POST http://http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

**Response:**
```json
{
  "prediction": 5,
  "probabilities": {
    "3": 0.01,
    "4": 0.05,
    "5": 0.35,
    "6": 0.40,
    "7": 0.15,
    "8": 0.03,
    "9": 0.01
  }
}
```

## Example Input Values

### Example 1 (Red Wine Characteristics)
```
Fixed Acidity: 7.4
Volatile Acidity: 0.7
Citric Acid: 0.0
Residual Sugar: 1.9
Chlorides: 0.076
Free Sulfur Dioxide: 11
Total Sulfur Dioxide: 34
Density: 0.9978
pH: 3.51
Sulphates: 0.56
Alcohol: 9.4
```

### Example 2 (White Wine Characteristics)
```
Fixed Acidity: 7.0
Volatile Acidity: 0.27
Citric Acid: 0.36
Residual Sugar: 20.7
Chlorides: 0.045
Free Sulfur Dioxide: 45
Total Sulfur Dioxide: 170
Density: 1.001
pH: 3.0
Sulphates: 0.45
Alcohol: 8.8
```

## Model Performance

The logistic regression model with SMOTE achieves:
- **Overall Accuracy**: ~48%
- **Balanced Predictions**: Better recall across all quality classes
- **Minority Class Improvement**: Significant improvement for rare quality scores
- **Quality Range**: Predictions span the full range (3-9)

### Performance Notes

- SMOTE improves minority class recall at the cost of overall accuracy
- Quality 3: 0% → 11% recall
- Quality 4: 2% → 26% recall
- Quality 8: 0% → 26% recall
- The model now makes more diverse predictions instead of favoring majority classes
- Trade-off: Lower overall accuracy but better representation of all quality levels

## Testing

Run the property-based tests:

```bash
pytest tests/ -v
```

The test suite includes:
- Feature scaling validation (mean ≈ 0, std ≈ 1)
- Correlation coefficient bounds ([-1, 1])
- 100 iterations per property test using Hypothesis

## Exploratory Data Analysis

View the EDA notebook for detailed analysis:

```bash
jupyter notebook notebooks/eda.ipynb
```

The notebook includes:
- Summary statistics for all features
- Quality score distribution analysis
- Feature distribution visualizations
- Correlation analysis with quality
- Red vs. white wine comparisons

## API Endpoints

### GET /
Returns the main input form page

### POST /predict
Accepts form data and returns HTML prediction result

### POST /api/predict
Accepts JSON data and returns JSON prediction
- **Input**: JSON object with 11 physicochemical features
- **Output**: JSON with prediction and probabilities

### GET /health
Health check endpoint
- **Output**: JSON with service status

## Quality Score Interpretation

- **3-4**: Low quality wine ⚠️
- **5-6**: Medium quality wine ✓
- **7-9**: High quality wine ⭐

Note: Quality scores are based on expert evaluations (median of at least 3 assessments)

## Troubleshooting

### Model files not found
**Error**: `Model files not found. Please train the model first`

**Solution**: Run `python train.py` to train and save the model

### Import errors
**Error**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**: Install dependencies with `pip install -r requirements.txt`

### Port already in use
**Error**: `Address already in use`

**Solution**: Change the port in `src/app.py` or kill the process using port 5001

## Contributing

This project was developed as part of a machine learning classification system. Contributions for improvements are welcome.

## License

This project uses the Wine Quality Dataset from the UCI Machine Learning Repository.

## Acknowledgments

- Dataset: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  Decision Support Systems, Elsevier, 47(4):547-553, 2009.

## Contact

For questions or issues, please open an issue in the repository.
