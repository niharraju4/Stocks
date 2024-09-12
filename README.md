
### README.md

```markdown
# Stock Price Analysis and Prediction

## Overview

This project aims to analyze and predict stock prices using historical data retrieved from the Alpaca API. The analysis includes calculating correlations, visualizing trends, and applying various machine learning models to predict stock price movements.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Retrieval](#data-retrieval)
- [Data Preparation](#data-preparation)
- [Analysis](#analysis)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Findings](#findings)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Installation

To run this project, you need to have Python installed on your system. You can install the required libraries using pip:

```bash
pip install alpaca_trade_api pandas matplotlib seaborn numpy scikit-learn scipy
```

## Usage

1. Replace `YOUR_API_KEY` and `YOUR_API_SECRET` with your Alpaca API key and secret in the script.
2. Run the script to retrieve and analyze stock data.

```bash
python stock_analysis.py
```

## Data Retrieval

The project uses the Alpaca API to retrieve historical stock data for a list of ticker symbols. The data includes daily closing prices, highs, lows, trade counts, opening prices, volumes, and VWAPs.

```python
import alpaca_trade_api as alpaca

# Replace YOUR_API_KEY and YOUR_API_SECRET with your Alpaca API key and secret
alpaca_api = alpaca.REST('PKU5LEIDZZV83Y5ENASI', 'nvzOwJpyvr73GTitTDyga0MkI2Qd6RsK4PmhItbu', api_version='v2')

# Set the ticker symbol and time frame
ticker = ['CVX', 'COP', 'EOG', 'PXD', 'SLB', 'PSX', 'MPC', 'VLO', 'HES', 'OXY', 'FANG', 'DVN', 'HAL','TLT','USO']
timeframe = "1Day"

# Set the start and end dates for the data
start_date = "2015-01-01T00:00:00-00:00"
end_date = "2023-10-01T00:00:00-00:00"

data = alpaca_api.get_bars(ticker, timeframe, start_date, end_date).df
df = pd.DataFrame(data)
df = df[['close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap', 'symbol']]
```

## Data Preparation

The retrieved data is prepared for analysis by creating separate dataframes for each stock and calculating percentage changes.

```python
dctStockDfs = {}
for stock in ticker:
    dfString = 'df' + stock
    data = df[df['symbol'] == stock]
    data = data[['close']]
    data = data.dropna()
    data = pd.DataFrame(data)
    data = data.rename(columns={'close': stock + 'close'})
    exec(f"df{stock} = data")
    dctStockDfs[f'df{stock}'] = data
```

## Analysis

The project analyzes stock price correlations and trends over time by creating scatterplots and correlation matrices.

```python
for i in range(20, 61, 5):
    for key1, df1 in dctStockDfs.items():
        dfCorr = pd.concat([df1, dfYvar], join='inner', axis=1)
        dfCorr = dfCorr.dropna()
        dfCorr = dfCorr.pct_change(i)
        dfCorr[f'Rolling{str(i)}Day'] = dfCorr.iloc[:, 1].rolling(i).corr(dfCorr.iloc[:, 0])
        rollingCorr = dfCorr.iloc[:, 1].corr(dfCorr.iloc[:, 2])
        dfCorr.iloc[:, 1] = dfCorr.iloc[:, 1].shift(-i)
        dfCorr = dfCorr.dropna()
        corr = dfCorr.iloc[:, 1].corr(dfCorr.iloc[:, 0])
        if abs(corr) >= 0.1:
            print(f'Pct Change Period is {i}')
            plt.show()
            print(f'Correlation between XOM and {key1}: {corr}\n')
            plt.figure(figsize=(8, 6))
            plt.scatter(dfCorr.iloc[:, 0], dfCorr.iloc[:, 1])
            plt.title(f'Scatterplot of {key1} vs XOMClose')
            plt.xlabel(dfCorr.columns[0])
            plt.ylabel('XOMClose')
            plt.show()
        plt.close()
        plt.clf()
        del(dfCorr)
```

## Machine Learning Models

The project applies various machine learning models, including SVM, Decision Tree, Random Forest, and KNN, to predict stock price movements.

```python
# Data Scaling, SVM Model Training, and Evaluation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)
print("Decision Tree Model:")
print("Accuracy:", accuracy_score(y_test, dt_y_pred))
print("Classification Report:")
print(classification_report(y_test, dt_y_pred))

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
print("Random Forest Model:")
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
print("Classification Report:")
print(classification_report(y_test, rf_y_pred))

# K-Nearest Neighbors (KNN) Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)
print("K-Nearest Neighbors (KNN) Model:")
print("Accuracy:", accuracy_score(y_test, knn_y_pred))
print("Classification Report:")
print(classification_report(y_test, knn_y_pred))
```

## Results

### Comparison of Model Accuracies

```python
model_accuracies = {
    "Model": ["SVM with RBF Kernel", "Decision Tree", "Random Forest", "K-Nearest Neighbors (KNN)", "SVM with Linear Kernel"],
    "Accuracy (35% change)": [0.7411, 0.7571, 0.8830, 0.8723, 0.6135],
    "Accuracy (60% change)": [0.7774, 0.7662, 0.8237, 0.8071, 0.7644]
}

# Create a DataFrame from the dictionary
df_accuracies = pd.DataFrame(model_accuracies)

# Print the comparison table
print("Comparison of Model Accuracies:")
print(df_accuracies)
```

### Best Models

```python
# Identify the best models based on accuracy
best_model_35 = df_accuracies.loc[df_accuracies['Accuracy (35% change)'].idxmax()]
best_model_60 = df_accuracies.loc[df_accuracies['Accuracy (60% change)'].idxmax()]

# Print the best models
print("\nBest Model for 35% Change Data:")
print(best_model_35)

print("\nBest Model for 60% Change Data:")
print(best_model_60)
```

## Findings

### The Random Forest model stands out as the best performer for predicting stock movements based on 35% change data. Its high accuracy can be attributed to its ability to handle nonlinear relationships and its robustness to overfitting, making it well-suited for complex stock market data. This suggests that Random Forest could be a reliable choice for developing trading strategies that require accurate predictions of stock price movements.

### Title: Analyzing the Correlation between Energy Sector Stocks over Different Time Periods

In this analysis, I explored the correlation between Exxon Mobil (XOM) and various energy sector stocks over different time periods ranging from 30 to 60 days. By calculating the percentage change and rolling correlations, I aimed to understand how closely these stocks move in relation to XOM. My findings indicate that certain stocks exhibit significant correlations with XOM, particularly over specific time periods. For instance, the correlation coefficients and corresponding p-values suggest that some stocks may have a stronger relationship with XOM during certain intervals. This information could be valuable for investors looking to diversify their portfolios within the energy sector or for those seeking to develop trading strategies based on the movement patterns of these stocks in relation to a major player like Exxon Mobil.

```python
# Create scatterplots and correlation matrices
for i in range(30, 61, 5):
    for key1, df1 in dctStockDfs.items():
        # Reset the index if necessary
        df1 = df1.reset_index(drop=True)
        dfYvar_aligned = dfYvar.reset_index(drop=True)

        # Concatenate the dataframes
        dfCorr = pd.concat([df1, dfYvar_aligned], join='inner', axis=1)
        dfCorr = dfCorr.dropna()  # Calculate percentage change
        dfCorr = dfCorr.pct_change(i)
        dfCorr[f'Rolling{str(i)}Day'] = dfCorr.iloc[:, 1].rolling(i).corr(dfCorr.iloc[:, 0])
        rollingCorr = dfCorr.iloc[:, 1].corr(dfCorr.iloc[:, 2])
        dfCorr.iloc[:, 1] = dfCorr.iloc[:, 1].shift(-i)
        dfCorr.iloc[:, 1] = (dfCorr.iloc[:, 1] > 0).astype(int)

        dfCorr = dfCorr.dropna()
        # Calculate and display correlation
        if not dfCorr.empty:
            correlation_coefficient, p_value = pointbiserialr(dfCorr.iloc[:, 0], dfCorr.iloc[:, 1])
            correlation_coefficient2, p_value2 = pointbiserialr(dfCorr.iloc[:, 0], dfCorr.iloc[:, 2])

            if p_value < 0.05:
                print(f'Correlation between XOM and {key1}: {correlation_coefficient}')
                print(f'P-value: {p_value}')
                print(f'Shape: {dfCorr.shape}')

                # Create scatterplot
                plt.figure(figsize=(8, 6))
                plt.scatter(dfCorr.iloc[:, 0], dfCorr.iloc[:, 1])
                plt.title(f'Scatterplot of {key1} vs XOMClose')
                plt.xlabel(dfCorr.columns[0])
                plt.ylabel('XOMClose')
                plt.show()
                plt.close()
                plt.clf()
                print(f'Pct Change Period is {i}')
                plt.show()
                print(f'Correlation between XOM and {str(i)}Day Rolling Corr of {key1}: {rollingCorr}\n')
                print(f'P-value: {p_value2}')
                print(f'Shape: {dfCorr.shape}')
                # Create scatterplot
                plt.figure(figsize=(8, 6))
                plt.scatter(dfCorr.iloc[:, 2], dfCorr.iloc[:, 1])
                plt.title(f'Scatterplot of {key1} vs XOMClose')
                plt.xlabel(dfCorr.columns[2])
                plt.ylabel('XOMClose')
                plt.show()
            else:
                plt.close()
                plt.clf()
            del(dfCorr)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.



## Author: Nihar Raju
```

