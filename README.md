# Paddy Image Analyzer

This project is a web application built with FastAPI and a simple HTML/JavaScript front-end that allows users to upload paddy images and get predictions for disease classification, variety classification, and age prediction.

## Project Structure

```
.
├── main.py
├── templates/
│   └── index.html
├── model_disease.keras
├── model_variety.keras
├── model_age.keras
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

**Note**: You will need to replace the placeholder model files (.h5 or .keras) with your actual trained models for each task.

## Prerequisites

- **Python 3.12+**: Ensure you have a compatible version of Python installed.

## Setup

**1. Clone the repository and navigate to the project directory:**

```sh
git clone https://github.com/nguyenpnguyen/paddy-predict.git
cd paddy-predict
```

**2. Install dependencies:**

```
pip install -r requirements.txt
```

## Running the Application

1. **Ensure your virtual environment is activated**

2. **Run the FastAPI application using `uvicorn`:**

```
uvicorn main:app --reload
```

The --reload flag is useful during development as it restarts the server automatically when you make changes to the code.

3. **Access at** `http://localhost:8000`
