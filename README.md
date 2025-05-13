# Paddy Image Analyzer

This project is a web application built with FastAPI and a simple HTML/JavaScript front-end that allows users to upload paddy images and get predictions for disease classification, variety classification, and age prediction.

## Project Structure

```
.
├── main.py
├── templates/
│   └── index.html
├── paddy_disease_model.keras
├── paddy_variety_model.keras
├── paddy_age_model.keras
├── pyproject.toml
├── uv.lock
└── README.md
```

**Note**: You will need to replace the placeholder model files (.h5 or .keras) with your actual trained models for each task.

## Prerequisites

- **Python 3.8+**: Ensure you have a compatible version of Python installed.
- **uv**: The uv package manager. If you don't have it, you can install it by following the instructions on the uv GitHub page. A common installation method is:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh 
```

Make sure uv is in your system's PATH.

## Setup

**1. Clone the repository and navigate to the project directory:**

```sh
git clone https://github.com/nguyenpnguyen/paddy-predict.git
cd paddy-predict
```

**2. Create a virtual environment using uv:**

```sh
uv venv
```

This creates a virtual environment named `.venv` in your project directory.

**3. Activate the virtual environment:**

- On macOS and Linux:

```
source .venv/bin/activate

```

- On Windows:

```
.venv\Scripts\activate

```

4. **Install dependencies using `uv`:**

```
uv pip sync
```

5. **Place your trained models:**
Put your trained model files (e.g., `paddy_disease_model.keras`, `paddy_variety_model.keras`, `paddy_age_model.keras`) in the root of the project directory, alongside `main.py`. Make sure the filenames match the `_MODEL_PATH` variables in `main.py`.

## Running the Application

1. **Ensure your virtual environment is activated** (see step 4 in Setup).

2. **Run the FastAPI application using `uvicorn`:**

```
uv run uvicorn main:app --reload
```

The --reload flag is useful during development as it restarts the server automatically when you make changes to the code.

3. **Access the front-end:**
Open your web browser and go to `http://127.0.0.1:8000`.
