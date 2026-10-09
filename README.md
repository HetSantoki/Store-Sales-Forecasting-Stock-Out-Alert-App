# Store Sales Forecasting Project

## Overview

This repository contains a **Store Sales Forecasting** application built with **Streamlit**. The app demonstrates how to load historical sales data, train a time‑series forecasting model, and visualise future sales predictions for multiple stores.

## Features

- **Data ingestion** for CSV files containing daily sales per store.
- **Forecasting models** using Prophet / ARIMA (configurable).
- Interactive **Streamlit dashboard** with:
  - Store selector
  - Date‑range picker
  - Real‑time line chart of historical and forecasted sales
  - Downloadable CSV of predictions
- Dark‑mode ready UI with modern, vibrant styling.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Store_Sales_Forecasting_Project.git
cd Store_Sales_Forecasting_Project

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install required packages
pip install -r requirements.txt
```

## Running the Dashboard

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

## Project Structure

```
Store_Sales_Forecasting_Project/
│   README.md            # ← This file
│   requirements.txt     # Python dependencies
│   app.py               # Main Streamlit app
│   model.pkl            # Forecasting model utilities
│   data/                # Sample data files
│   images/              # Dashboard screenshots (add yours here)
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements, new models, or UI enhancements.
