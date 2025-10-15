# assetclustering

About:
This program clusters assets based on daily price changes taken from yfinance API. It clusters assets using the K-medoids algorithm with Dynamic Time Warp to account for misaligned time series.
You can select different k for clustering, perform analysis on optimal k for k-medoids with the visualizations for elbow method and silhouette scores, and visualize optimal clusterings for a 
specific K.

Dependencies:
- yfinance
- numpy
- pandas
- matplotlib.pyplot
- matplotlib.widgets
  
*Ensure dependencies are installed for python otherwise code will not work

Usage Instructions:
1. Open AssetClustering.py
2. Inside the code, adjust GLOBAL variables as needed
3. Inside the code, adjust the list 'tickers' as needed
4. Run the code and follow the prompts
