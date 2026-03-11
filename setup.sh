#!/bin/bash

# Setup script for Correlation Analysis Dashboard

echo "=================================================="
echo "  Correlation Analysis Dashboard - Setup"
echo "=================================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✅ Python 3 found"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt --break-system-packages

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Generate sample data
echo "📊 Generating sample dataset..."
python3 generate_sample_data.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate sample data"
    exit 1
fi

echo "✅ Sample data generated"
echo ""

# Run examples
echo "🎯 Running example analyses..."
python3 examples.py

echo ""
echo "=================================================="
echo "  ✅ Setup Complete!"
echo "=================================================="
echo ""
echo "To launch the dashboard, run:"
echo "  python3 main.py sample_data.csv"
echo ""
echo "Or explore examples:"
echo "  python3 examples.py 1  # Basic analysis"
echo "  python3 examples.py 2  # Pattern detection"
echo "  python3 examples.py 3  # Custom visualization"
echo "  python3 examples.py 4  # Filtered analysis"
echo "  python3 examples.py 5  # Export results"
echo "  python3 examples.py 6  # Launch dashboard"
echo ""
