import yfinance as yf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Slider

'''
Overview:
This program fetches daily asset prices over a given period from yfinance api and clusters assets by their z-score normalized log price change
time series using k-medoids with dynamic time warping (DTW) to capture similar shapes under misalignment. Additionally, this program allows
for the analysis of an optimal k for k-medoids by visualizing the elbow method and silhouette coefficient for values of k, as well as the visualization
of clusters formed by k-medoids.
'''

'''Variable Properties, change global variable or tickers as desired'''

#Global variables to adjust
GLOBAL_KMEDOID_SAMPLES = 5000   #Number of samples for k-medoids (Reduce samples if program is slow)
GLOBAL_W_FACTOR = 0.05          #Factor to calculate sakoe band half-width
GLOBAL_PRICE_PERIOD = "1y"      #Period of prices for assets
GLOBAL_LOG_DATA = True          #Take log of data
GLOBAL_AGGREGATE_WEEKLY = False #Aggregate data for weekly changes (enable if program is slow due to too much data)

'''ADJUST LIST OF TICKERS HERE'''
tickers = ["KO", "PEP", "SPGI", "MCO", "MSCI", "META", "MSFT", "GOOGL"]

'''**********************FUNCTIONS********************************'''

'''getDailyChange(): 
DESC: Takes in daily asset prices, returns daily price changes.
INPUT: np.array daily asset prices, bool log, bool zscore
OUTPUT: returns modified nparray containing z-scored log daily changes by default
'''
def getDailyChange(prices, log=GLOBAL_LOG_DATA, zscore=True):
    prices = np.asarray(prices, dtype=np.float32) #Create array as type float32 to save memory
    prices = prices[~np.isnan(prices)]  #Remove nan cols
    #Check if prices is empty
    if len(prices) < 2:
        return np.array([], dtype=np.float32)

    #Get log daily change/daily change
    if log:
        changes = np.log(prices[1:] / prices[:-1])
    else:
        changes = (prices[1:] - prices[:-1]) / prices[:-1]

    #Remove nan cells
    changes = changes[np.isfinite(changes)]

    #z-score (log) daily changes
    if zscore and len(changes) > 1:
        mean = np.mean(changes)
        std = np.std(changes)
        if std > 0:
            changes = (changes - mean) / std
        else:
            changes = changes - mean  # handle flat data case

    return changes.astype(np.float32)

'''getDTWmat(): 
DESC: returns the dynamic time warping (dtw) matrix (cost matrix) to calculate distance/cost between assets inputted
INPUT: list of ticker symbols
OUTPUT: returns nparray filled with dtw cost between asset in ticker[i] and asset in ticker[j]
'''
def getDTWmat(assets):
    n=len(assets)

    #prepare asset data by converting to log and z-score norm
    assets_daily_changes = {}
    for i in range(len(assets)):
        assets_daily_changes[i] = getDailyChange(np.array(data[assets[i]].values)) #Get normalized log price change for asset
        print("Getting Prices for asset ", assets[i])

    #initialize dtw matrix
    dtw_mat = np.zeros((n,n))

    #calculate dtw matrix while keeping track in counter for progress tracker
    counter = 1
    for i in range(n):
        for j in range(n):
            print("Calculating DTW Cost between ", assets[i], " and ", assets[j])
            dtw_mat[i][j] = DTW(assets_daily_changes[i],assets_daily_changes[j], True)
            print("Done: ", counter, " / ", n*n)
            counter += 1
    
    return dtw_mat

'''DTW():
DESC: Performs Dynamic Time Warping algorithm on 2 time series to calculate cum. cost of aligning the two series, returning either cum. cost matrix and objective value,
or just objective value. Includes sakoe-chiba band to optimize performance for larger time series (for example 5y) by only considering points some range
dictated by w away from the matrix's diagonal.
INPUT: time series ts1, time series ts2, bool cost_b (if true returns objective val, else returns cost matrix and objective val), sakoe band half width
OUTPUT: if cost_b true returns objective val, else returns cost matrix and objective val
'''
def DTW(ts1, ts2, cost_b = False, w=None):

    #Check empty time series
    if len(ts1) == 0 or len(ts2) == 0:
        return math.inf if cost_b else (np.empty((0, 0), dtype=np.float32), math.inf) #Return empty 0x0 matrix and obj value = inf

    #Initialize time series
    ts1 = np.asarray(ts1, dtype=np.float32)
    ts2 = np.asarray(ts2, dtype=np.float32) #Define matrix size variables
    N, M = ts1.size, ts2.size

    #sakoe band
    if w is None:
        w = int(GLOBAL_W_FACTOR*N)            # default w
    w = max(w, abs(N - M))       # ensure bounds

    #Initialize cost matrix
    INF = np.inf
    c_mat = np.full((N + 1, M + 1), INF, dtype=np.float32)
    c_mat[0, 0] = 0.0

    #Calculate cost matrix
    for i in range(N):
        a = ts1[i]
        #Get sakoe band indices
        j_start = max(0, i - w)
        j_end   = min(M - 1, i + w)
        for j in range(j_start, j_end + 1):
            #Track 3 surrounding cells for calculation
            d = abs(a - ts2[j])

            #get min penalty, compare w/ m1
            m1 = c_mat[i,j] #diag = 0
            m2 = c_mat[i, j+1] #up = 1
            m3 = c_mat[i+1, j] #left = 2

            if m2 < m1: m1 = m2
            if m3 < m1: m1 = m3

            c_mat[i + 1, j + 1] = d + m1
    
    total_cost = float(c_mat[N, M])

    if cost_b:
        return total_cost

    return c_mat[1:, 1:], total_cost

'''kmedoids():
DESC: Performs the k-medoids algorithm using DTW as the cost/distane metric between points, initializing k medoids randomly with uniform distribution.
INPUT: list of tickers assets, int k, dtw_mat = dynamic time warp matrix, if val = false return cluster, else return objective value
OUTPUT: if val = false return cluster, else return objective value and cluster
'''
def kmedoids(assets, k, dtw_mat, val = False):

    #Initialize tracking variables
    min_cluster = [[] for _ in range(k)]
    min_medoids = []
    min_obj_val = math.inf

    n = len(assets)

    #Check k > n (Medoids > number of stocks)
    if k > n:
        raise ValueError("k > n")
    
    medoids = []

    #Randomly select medoids
    while len(medoids) < k:
        medoid = random.randint(0,n-1)
        if medoid not in medoids:
            medoids.append(medoid)

#Kmedoids
    visited = set() #set of visited ordered tuples
    max_loop = 100

    for _ in range(max_loop):
        combination = tuple(sorted(medoids))
        if combination in visited:  #Check for visited combination
            break #stop if we reach same combination
        visited.add(combination)

        clusters = [[] for _ in range(k)]
        temp_obj_val = 0

        #Assignment step
        for i in range(n):
                min_cost = math.inf
                min_m_i = None
                #Compare asset w/ each medoid
                for j in range(len(medoids)):
                    temp_cost = dtw_mat[i, medoids[j]]

                    if temp_cost < min_cost:
                        min_cost = temp_cost
                        min_m_i = j
                #Place asset into cluster based on medoid
                if min_m_i is None:
                    continue  # skip asset if all distances are inf
                clusters[min_m_i].append(i)
                temp_obj_val += min_cost  
        
        if temp_obj_val < min_obj_val:
            min_cluster = [cl[:] for cl in clusters]
            min_obj_val = temp_obj_val
            min_medoids = medoids[:]
        
        #Update medoids
        new_medoids = medoids[:]
        for i in range(k):
            cluster = clusters[i]
            if not cluster: #cluster empty
                continue
            
            sub_d = dtw_mat[np.ix_(cluster, cluster)] #distances between all points in cluster
            min_sub_medoid = np.argmin(sub_d.sum(axis=1)) #Get argmin of sums of distances for all points in cluster (rows)
            new_medoids[i] = cluster[min_sub_medoid] #Add min distance point to medoid
        

        if new_medoids == medoids:
            break
        medoids = new_medoids

    if (val):
        return min_obj_val, min_cluster
    else:
        return min_cluster


'''kmedoids_Sample():
DESC: Iterates through x number of samples of k-medoids to account for randomness of selecting initial medoid, and returning the most optimal clustering for k out of x samples
INPUT: ticker list, int k, dtw_mat, if val = false return cluster, else return objective value, int samples
OUTPUT: if val = false return optimal cluster from samples, else return objective value and optimal cluster
'''
#Sample K-medoids wrapper
def kmedoids_Sample(assets, k, dtw_mat, val = False, samples = GLOBAL_KMEDOID_SAMPLES):
    min_obj = math.inf
    min_cluster = None

    #Store obj val and cluster in variables, update if new min is found
    for _ in range(samples):
        obj_val, cluster = kmedoids(assets, k, dtw_mat, val=True)
        if obj_val < min_obj:
            min_obj = obj_val
            min_cluster = cluster
    
    if val:
        return min_obj, min_cluster
    else:
        return min_cluster

'''calcSilhouette():
DESC: Calculates silhouette coefficient for each point in each cluster, returns silhouette score of clustering
INPUT: Optimal clusters, dtw_mat
OUTPUT: float silhouette score
'''
def calcSilhouette(clusters, dtw_mat):
    s_vals = []

    # iterate through clusters
    for cluster in clusters:
        for p in cluster:  # iterate current point

            #calculate a: avg distance to other points in cluster
            a = 0
            for j in cluster:
                if p != j:
                    a += dtw_mat[p, j]
            if len(cluster) > 1:
                a /= len(cluster) - 1
            else:
                a = 0

            #calculate b: min avg distance to other points outside cluster
            b = math.inf
            for other_cluster in clusters:
                if other_cluster == cluster or not other_cluster:
                    continue
                temp = 0
                for j in other_cluster:
                    temp += dtw_mat[p, j]
                temp /= len(other_cluster)
                if temp < b:
                    b = temp

            #calc silhouette score
            denom = max(a, b)
            s = 0
            if denom != 0:
                s = (b - a) / denom
            s_vals.append(s)

    # average silhouette across all points
    if s_vals:
        return sum(s_vals) / len(s_vals) 
    else:
        return 0

'''printMinCluster(clusters):
DESC: Helper function to print clusters neatly
INPUT: Cluster to print
'''
def printMinCluster(clusters):
    for i in range(len(clusters)):
        print(f"Cluster {i}:")
        for j in clusters[i]:         # iterate through elements inside that cluster
            print(f"  {tickers[j]}")  # print the ticker corresponding to that index

'''plot_clusters_scrollable(clusters):
DESC: Function plots clusters according to tickers and data given. Data is by default z-scored log daily change of asset
INPUT: cluster to plot, ticker list, raw price data
'''
def plot_clusters_scrollable(clusters, tickers, data):
    if data is None or data.empty:
        raise ValueError("Price data is empty.")

    # keep only non-empty clusters
    valid_clusters = [c for c in clusters if c]
    if not valid_clusters:
        raise ValueError("All clusters are empty.")

    n_clusters = len(valid_clusters)

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.subplots_adjust(bottom=0.25)
    slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(slider_ax, 'Cluster', 0, n_clusters - 1, valinit=0, valstep=1)

    def update(_):
        ax.clear()
        ci = int(slider.val)
        idxs = valid_clusters[ci]

        plotted_any = False
        for i in idxs:
            if i < 0 or i >= len(tickers):
                continue
            tkr = tickers[i]
            if tkr not in data.columns:
                continue

            # clean data
            s = data[tkr].dropna()
            if s.shape[0] < 2:
                continue
            
            # calculate z-score normalized log daily changes
            changes = getDailyChange(s.values, log=True, zscore=True)  # z-score normalized log returns
            if changes.size == 0:
                continue

            # align index: length n-1 after returns
            idx = s.index[1:1 + len(changes)]
            ax.plot(idx, changes, label=tkr, linewidth=1.2)
            plotted_any = True

        # labels/titles
        period_str = "Weekly " if GLOBAL_AGGREGATE_WEEKLY else "Daily "
        ax.set_title(f"Cluster {ci} — Z-scored Log Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel(period_str + "Z-scored Log Return")
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(fontsize=8)
        plt.draw()

    update(0)
    slider.on_changed(update)
    plt.show()

'''plot_k_elbow_and_silhouette():
DESC: Plots elbow graph of k using objective value and silhouette coefficient graph for k for analysis of optimal k
INPUT: ticker list, max int k to plot, dtw_mat, int number of samples
'''
def plot_k_elbow_and_silhouette(assets, max_k, dtw_mat, samples=GLOBAL_KMEDOID_SAMPLES):
    n = len(assets)
    if max_k > n:
        max_k = n  # cap at n

    ks_elbow = list(range(1, max_k + 1))
    ks_sil   = list(range(2, max_k + 1))  # silhouette undefined for k<2

    obj_vals = []
    sil_vals = []

    # k = 1 (elbow only)
    min_obj, _ = kmedoids_Sample(assets, 1, dtw_mat, val=True, samples=samples)
    obj_vals.append(min_obj)
    print("k = 1 complete — obj =", min_obj)

    # k >= 2: compute both objective and silhouette from the SAME clustering run
    for k in ks_sil:
        min_obj, clusters_k = kmedoids_Sample(assets, k, dtw_mat, val=True, samples=samples)
        obj_vals.append(min_obj)
        s = calcSilhouette(clusters_k, dtw_mat)
        sil_vals.append(s)
        print(f"k = {k} complete — obj = {min_obj:.4f}, silhouette = {s:.4f}")

    # pick best k by silhouette (non-trivial)
    best_idx = int(np.argmax(sil_vals))
    best_k = ks_sil[best_idx]

    # --- plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    # Elbow (objective)
    ax1.plot(ks_elbow, obj_vals, marker='o', linestyle='-')
    ax1.set_xlabel("k (Number of Clusters)")
    ax1.set_ylabel("Total Within-Cluster Cost (Objective)")
    ax1.set_title("Elbow Method (DTW K-Medoids)")
    ax1.grid(True, alpha=0.3)

    # Silhouette
    ax2.plot(ks_sil, sil_vals, marker='o', linestyle='-')
    ax2.scatter([best_k], [sil_vals[best_idx]], s=80)
    ax2.set_xlabel("k (Number of Clusters)")
    ax2.set_ylabel("Average Silhouette (DTW)")
    ax2.set_title("Silhouette vs k (DTW K-Medoids)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "best_k_silhouette": best_k,
        "ks_elbow": ks_elbow, "obj_vals": obj_vals,
        "ks_silhouette": ks_sil, "sil_vals": sil_vals
    }

'''cleanData():
DESC: Prepares data by removing bad columns and resampling if necessary
INPUT: raw price data, ticker list, bool weekly (if true aggregate for weekly data, else use daily)
OUTPUT: Return cleaned data, list of cleaned tickers
'''
def cleanData(data, tickers, weekly = GLOBAL_AGGREGATE_WEEKLY):
    before_cols = set(data.columns)
    data = data.dropna(axis=1, how="all")
    after_cols = set(data.columns)
    removed_tickers = list(before_cols - after_cols)
    tickers = [t for t in tickers if t not in removed_tickers]

    print ("removed due to full NaN: ", removed_tickers)

    #Resample to weekly prices
    if weekly:
        data = data.resample("W-FRI").last()
    return data, tickers

'''make_synthetic_prices():
DESC: Overwrites data with 3 distinct time series patterns for testing clustering.
INPUT: raw price changes, random seed
OUTPUT: returns synthetic data of same size
'''
def make_synthetic_prices(data, seed=42):
    rng  = np.random.default_rng(seed)
    idx  = data.index
    cols = list(data.columns)
    T, n = len(idx), len(cols)

    #split into 3 partitions
    groups = np.array_split(np.arange(n), 3)

    #time initialization
    t = np.linspace(0, 3*np.pi, T)

    #container
    rets = np.zeros((T, n), dtype=np.float64)

    #sin
    for j in groups[0]:
        phase = rng.uniform(-np.pi/4, np.pi/4)
        rets[:, j] = 0.010 * np.sin(t + phase) + rng.normal(0, 0.004, T)

    #cos
    for j in groups[1]:
        phase = rng.uniform(-np.pi/4, np.pi/4)
        rets[:, j] = 0.010 * np.cos(t + phase) + rng.normal(0, 0.004, T)

    #white noise
    for j in groups[2]:
        rets[:, j] = rng.normal(0, 0.012, T)

    #convert returns to prices
    prices = np.zeros((T, n), dtype=np.float64)
    for j, col in enumerate(cols):
        # Use the first non-NaN of the original as starting price
        start = float(data[col].dropna().iloc[0]) if data[col].notna().any() else 100.0
        logp0 = np.log(start)
        logp  = logp0 + np.cumsum(rets[:, j])
        prices[:, j] = np.exp(logp)

    return pd.DataFrame(prices, index=idx, columns=cols)

'''
**************Program Code******************
'''

#Initialization

#Get data from yf
data = yf.download(tickers, period=GLOBAL_PRICE_PERIOD, interval = "1d")["Close"]

#clean data
data, tickers = cleanData(data, tickers)

#OVERWRITE w/SYNTHETIC DATA (uncomment below line for testing)
#data = make_synthetic_prices(data)

#Calculate DTW matrix (Most expensive)
dtw_mat = getDTWmat(tickers)

#Program loop
while True:
    try:
        user_k = int(input("Enter k value for k-medoids (or 0 to view k, negative integer to exit): "))
        if user_k < 0:
            print("Exit")
            break
        if user_k == 0:
            plot_k_elbow_and_silhouette(tickers, int(len(tickers)/2), dtw_mat)
            continue

        # Run clustering
        clusters = kmedoids_Sample(tickers, user_k, dtw_mat)
        print(f"\n--- Results for k = {user_k} ---")
        printMinCluster(clusters)

        # Plot the resulting clusters
        plot_clusters_scrollable(clusters, tickers, data)

    except ValueError:
        print("Enter valid integer.")
    except KeyboardInterrupt:
        print("\nStopped")
        break



