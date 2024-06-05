import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Excel sheets
file_path = 'path_to_your_excel_file.xlsx'
sheet_names = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4']

# Load all sheets into a dictionary of DataFrames
data = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names}

# Example of accessing a specific sheet
df1 = data['Sheet1']
df2 = data['Sheet2']
df3 = data['Sheet3']
df4 = data['Sheet4']

# Preprocess data (example: handle missing values, normalize data)
for name, df in data.items():
    df.fillna(df.mean(), inplace=True)  # Simple example: fill missing values with mean

# Regression Analysis
def regression_analysis(df, x_column, y_column):
    X = df[x_column]
    y = df[y_column]
    X = sm.add_constant(X)  # Adds a constant term to the predictor

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

    plt.figure(figsize=(10, 6))
    plt.scatter(X[x_column], y, label='Data')
    plt.plot(X[x_column], predictions, color='red', label='OLS Regression')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.title('Regression Analysis')
    plt.show()

# Perform regression analysis on Sheet1
regression_analysis(df1, 'independent_variable', 'dependent_variable')

# Time Series Analysis
def time_series_analysis(df, date_column, value_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)

    decomposition = sm.tsa.seasonal_decompose(df[value_column], model='additive')
    decomposition.plot()
    plt.show()

    model = sm.tsa.ARIMA(df[value_column], order=(5, 1, 0))
    results = model.fit()
    print(results.summary())

    df['forecast'] = results.predict(start=len(df), end=len(df) + 12, dynamic=True)
    df[[value_column, 'forecast']].plot(figsize=(12, 8))
    plt.show()

# Perform time series analysis on Sheet2
time_series_analysis(df2, 'date_column', 'value_column')

# Cluster Analysis
def cluster_analysis(df, n_clusters):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_scaled)
    df['cluster'] = kmeans.labels_

    silhouette_avg = silhouette_score(df_scaled, kmeans.labels_)
    print(f'Silhouette Score: {silhouette_avg}')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue='cluster', data=df, palette='viridis')
    plt.title('Cluster Analysis')
    plt.show()

# Perform cluster analysis on Sheet3
cluster_analysis(df3, 3)

# Summary statistics for each sheet
for name, df in data.items():
    print(f'\nSummary statistics for {name}:')
    print(df.describe())

# Example of combining data from multiple sheets
combined_df = pd.concat([df1, df2, df3, df4], axis=1)
print(combined_df.head())
