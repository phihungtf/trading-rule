---
transition: fade
---

## Getting data from public sources

We will use Yahoo Finance as our data source for price data.

For example, to retrieve the stock data of Apple Inc. (AAPL) from December 31, 2019, to December 31, 2020, on a daily basis, we can use the following URL:

https://finance.yahoo.com/quote/AAPL/history?period1=1577811600&period2=1609433999&filter=history&interval=1d&frequency=1d

---
layout: image
image: "/aapl-stock-data.png"
---

---
transition: fade
---

### Create a class to get stock data

```python {all|1,2,12-18}
class YahooDailyReader():
    def __init__(self, symbol=None, start=None, end=None):
        self.symbol = symbol
        self.start = start
        self.end = end

        # Convert start and end dates to Unix timestamp format.
        unix_start = int(time.mktime(self.start.timetuple()))
        day_end = self.end.replace(hour=23, minute=59, second=59)
        unix_end = int(time.mktime(day_end.timetuple()))

        # Build URL to get data from Yahoo Finance.
        url = 'https://finance.yahoo.com/quote/{}/history?'
        url += 'period1={}&period2={}'
        url += '&filter=history'
        url += '&interval=1d'
        url += '&frequency=1d'
        self.url = url.format(self.symbol, unix_start, unix_end)
```

---

### Create a class to get stock data

```python {1-4|6-7|9-10|12-13|15-16|18-20|22-23|all}
class YahooDailyReader():
    def read(self):
        # Download data from Yahoo Finance.
        stock_data = yf.download(self.symbol, start=self.start, end=self.end, interval='1d')

        # Convert JSON data to a pandas DataFrame.
        df = pd.DataFrame(stock_data).reset_index()

        # Add the stock symbol as a column in the DataFrame.
        df.insert(1, 'symbolid', self.symbol)

        # Convert Unix timestamps to date objects.
        df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.date

        # Drop rows where 'Close' price is NaN.
        df = df.dropna(subset=['Close'])

        # Rename columns.
        colnames = ['Date', 'symbolid', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
        df.columns = colnames

        # Set 'Date' column as index and return the DataFrame.
        return df.set_index('Date')
```

---

### Use our class to get APPL stock data

```python
symbol = 'AAPL'
start = datetime(1990, 1, 1)
end = datetime(2024, 5, 28)
reader = YahooDailyReader(symbol=symbol, start=start, end=end)
df = reader.read()
df.head()
```

![AAPL stock sample](/aapl-stock-sample.png)

---

### Download list of NASDAQ stock symbols

Go to the following URL to download a list of NASDAQ stock symbols (we will use _Mega_, _Large_ and _Medium_ for our Market Cap filter):

https://www.nasdaq.com/market-activity/stocks/screener

![NASDAQ stock screener](/nasdaq-stock-screener.png)

---

### Download data for all NASDAQ stocks

```python {1-3|5-6|8-12,17|10,16|17,20|22-25}
# Read the list of stock symbols from a CSV file.
symlist = pd.read_csv('csvdata/descdata.csv')
symlist = symlist['Symbol'].tolist()

# List of error symbols.
error_syms = []

# Get historical data for all stock symbols.
for i, sym in enumerate(symlist):
    try:
        reader = YahooDailyReader(symbol=sym, start=start, end=end)
        df = reader.read()

        print(f'{i+1}/{len(symlist)}: {sym} - {df.shape[0]} rows')

        df.to_csv(f'csvdata/{sym}.csv')
    except Exception as e:
        print(f'{i+1}/{len(symlist)}: {sym} - error: {e}')

        error_syms.append(sym)

# Remove error symbols from the list and update the csvdata/descdata.csv file.
symlist = [sym for sym in symlist if sym not in error_syms]
symlist = pd.DataFrame({'Symbol': symlist})
symlist.to_csv('csvdata/descdata.csv', index=False)
```

---
layout: image
image: "/crawling-data.png"
---

---
layout: image
image: "/data-crawled.png"
---

---

## Setting up the data

### Hypothesis formulation and in-sample testing

Starting by define numbers of days for holding the stock

```python
holding_days = 30
```

---

### Hypothesis formulation and in-sample testing

The `long_returns` function calculates the return of a long position for a stock over a specified holding period (number of days). This function assumes that the trader has no skill and will buy at the highest price of the day and sell at the lowest price of the day when closing the position.

```python
def long_returns(df, numdays):
    # Calculate the buy return by buying at the previous day's high and selling at the current day's low
    df['buyret'] = (df.Low / df.High.shift(numdays) - 1) * 100
    # Fill NaN values in the 'buyret' column with 0
    df.buyret.fillna(0, inplace=True)
    # Return the modified DataFrame
    return df
result = long_returns(df, holding_days)
result.tail()
```

![Long returns](/long-returns.png)

---

### Hypothesis formulation and in-sample testing

The `short_returns` function calculates the return of a short position for a stock over a specified holding period (number of days). This function assumes that the trader has no skill and will sell at the lowest price of the day and buy back at the highest price of the day when closing the position.

```python
def short_returns(df, numdays):
    # Calculate the sell return by selling at the lowest price of the day 'numdays' ago and buying back at the highest price of the day
    df['sellret'] = (df.Low.shift(numdays) / df.High - 1) * 100
    # Điền các giá trị NaN trong cột 'sellret' bằng 0 để tránh lỗ hổng dữ liệu
    df.sellret.fillna(0, inplace=True)
    # Return the modified DataFrame, including the new 'sellret' column
    return df
result = short_returns(df, holding_days)
result.tail()
```

![Short returns](/short-returns.png)

---

### Hypothesis formulation and in-sample testing

Labeling the data based on the conditions of long returns and short returns. If the long return is greater than 0.5%, we need to buy. If the short return is greater than 0.5%, we need to sell. The 0.5% threshold is used to avoid trades that may result in losses due to transaction costs. This threshold can be adjusted, depending on the trading efficiency of an investment fund.

```python
# label_data(df) label the data based on the conditions of long returns and short returns
def label_data(df):
    # Initialize the 'Action' column with a default value of 'None'
    df['Action'] = 'None'

    # If the long return is greater than 0.5%, set the action to 'Buy'
    df.loc[df['buyret'] > 0.5, 'Action'] = 'Buy'

    # If the short return is greater than 0.5%, set the action to 'Sell'
    df.loc[df['sellret'] > 0.5, 'Action'] = 'Sell'

    # Return the labeled DataFrame
    return df

result = label_data(df)
result.sample(5)
```

---

### Hypothesis formulation and in-sample testing

The result so far

![Labeled data](/labeling-data.png)

---

### Hypothesis formulation and in-sample testing

The `moving_avg_data` function calculates moving averages for the stock data.

```python {all|2-5|7-10|all}
def moving_avg_data(df, mavnames, mavdays):
    # Check if the length of the list of variable names and the number of days match
    if len(mavnames) != len(mavdays):
        print('Variable Names and Number of days must match')
        return

    # Loop through each variable name and number of days to calculate moving averages
    for i in range(len(mavnames)):
        # Calculate moving average for 'AdjClose' column with rolling window as the corresponding number of days
        df[mavnames[i]] = df.AdjClose.rolling(window=mavdays[i]).mean()

    # Return the DataFrame with the moving average columns added
    return df

mavnames = ['MA_20', 'MA_50']
mavdays = [20, 50]

result = moving_avg_data(df, mavnames, mavdays)
result.tail()
```

---

### Hypothesis formulation and in-sample testing

The result

![Moving averages](/moving-avg-data.png)

---

Now we can create datasets for training and testing

```python
def create_datasets(csvfilename, sample_size):
    # Randomly select indices from the list of stock symbols
    test_num = random.sample(range(0, len(symlist) - 1), sample_size)
    # Initialize a DataFrame to store the data
    data = pd.DataFrame()
    # Loop through each randomly selected index
    for i in range(0, len(test_num)):
        # Read data from the corresponding .csv file for the stock
        filename = 'csvdata/' + symlist.Symbol[test_num[i]] + '.csv'
        temp = pd.read_csv(filename)
        mavnames = ['mav5', 'mav10', 'mav20', 'mav30', 'mav50', 'mav100', 'mav200']
        mavdays = [5, 10, 20, 30, 50, 100, 200]
        # Calculate buy and sell returns based on previous data
        fwdret = 30 # Assume a future time period for calculating returns
        temp = moving_avg_data(label_data(short_returns(long_returns(temp, fwdret), fwdret)), mavnames, mavdays)
        # Drop rows containing NaN values
        temp=temp.dropna()
        # Merge the data into the main DataFrame
        data = pd.concat([data, temp])

    # Save the DataFrame to a .csv file
    data.to_csv('sampledata/' + csvfilename)
    print(csvfilename + ' written to disk')
```

---

### Hypothesis formulation and in-sample testing

```python
# Read the list of symbols file
symlist = pd.read_csv('csvdata/descdata.csv')

# Create training and testing datasets
create_datasets('train_50.csv', 50)
create_datasets('test_50.csv', 50)
```

![Training and testing datasets](/create-datasets.png)
