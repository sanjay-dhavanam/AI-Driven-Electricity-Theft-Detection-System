# Electricity Theft Detection System - Django

## Overview
A Django-based system for detecting electricity theft using adaptive machine learning models. This system analyzes half-hourly consumption patterns and identifies suspicious behavior.

## Features
- **User Portal**: View consumption data and anomaly results
- **Admin Panel**: Manage meters, trigger adaptive learning
- **Data Import**: Import CSV block files into SQLite database
- **Adaptive Learning**: Retrain models on recent data
- **Visualization**: Charts for consumption patterns

## Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd electricity_theft_django

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt