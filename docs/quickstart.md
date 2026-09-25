# Quick Start

## Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the forecast pipeline
python -m src.pipeline --config config/default.yaml
```

## Running the Dashboard

```bash
cd frontend-react
npm install
npm run dev
```

Open http://localhost:5173 to view the forecast dashboard.
