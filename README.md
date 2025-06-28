# Real-Time-Stock-Market-Prediction-using-GPU-Enabled-Hybrid-Model
#  Real-Time Stock Market Prediction using GPU-Enabled Hybrid Model (CNN + LSTM)

This project predicts real-time stock prices using a GPU-enabled hybrid deep learning model. It combines **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** layers to analyze both **image-based** historical patterns and **real-time numerical stock data**.

##  Features
- CNN + LSTM hybrid model
- Uses real-time stock data via `yfinance`
- Runs on GPU (Google Colab / CUDA enabled)
- Supports 2025 stock price prediction
- Autoencoder model for feature compression (if used)
- Plots results in clear, labeled graphs

##  Technologies Used
- Python
- TensorFlow / Keras
- yfinance
- NumPy, Pandas, Matplotlib
- CUDA (for GPU acceleration)
- Google Colab (or VS Code + Jupyter)

##  Project Structure
main.py / notebook.ipynb
├── requirements.txt
├── README.md
├── model.h5
├── output_graph.png
└── utils.py


##  Sample Output

![Output Graph](output_graph.png)

##  How to Run

```bash
pip install -r requirements.txt
python main.py

Dataset
Real-time data: yfinance API

Historical data: Yahoo Finance
