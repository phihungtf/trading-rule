---
transition: fade
---

## Getting data from public sources

We will use Yahoo Finance as our data source for price data.

For example, to retrieve the stock data of Apple Inc. (AAPL) from December 31, 2019, to December 31, 2020, on a daily basis, we can use the following URL:

https://finance.yahoo.com/quote/AAPL/history?period1=1577811600&period2=1609433999&filter=history&interval=1d&frequency=1d

<img src='/Yahoo!_Finance_logo_2021.png' className='mx-auto w-100 mt-10'/>

<!--
Chúng ta sẽ sử dụng Yahoo Finance làm nguồn dữ liệu. Ngoài ra, còn các nguồn dữ liệu khác như Google Finance, Alpha Vantage, Quandl, v.v. 
Dưới đây là URL để lấy dữ liệu giá cổ phiếu của Apple Inc. (AAPL) từ ngày 31 tháng 12 năm 2019 đến ngày 31 tháng 12 năm 2020 theo ngày.
-->

---
layout: image
image: "/aapl-stock-data.png"
---

<!--
Có 7 cột dữ liệu mà ta cần quan tâm là: 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
- 'Date': Ngày giao dịch.
- 'Open': Giá mở cửa.
- 'High': Giá cao nhất.
- 'Low': Giá thấp nhất.
- 'Close': Giá đóng cửa.
- 'Adj Close': Giá đóng cửa điều chỉnh.
- 'Volume': Khối lượng giao dịch.
-->

---
transition: fade
---

### Create a class to get stock data

```python {all}
class YahooDailyReader():
    def __init__(self, symbol=None, start=None, end=None):
        self.symbol = symbol
        self.start = start
        self.end = end
```

<!--
Chúng ta sẽ tạo một class để lấy dữ liệu giá cổ phiếu từ Yahoo Finance. Class này sẽ có 3 thuộc tính là 'symbol', 'start', 'end' để lưu mã cổ phiếu, ngày bắt đầu và ngày kết thúc.
-->

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

<!--
Chúng ta sẽ dùng thư viện yahoo finance để lấy dữ liệu giá cổ phiếu từ Yahoo Finance. Sau đó, chuyển dữ liệu JSON thành DataFrame của pandas. Tiếp theo, thêm cột 'symbolid' chứa mã cổ phiếu vào DataFrame. Chuyển đổi Unix timestamps thành đối tượng ngày. Loại bỏ các dòng có giá 'Close' là NaN. Đổi tên các cột và đặt cột 'Date' làm chỉ số của DataFrame.
-->

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

<!--
Thử lấy dữ liệu giá cổ phiếu của Apple Inc. (AAPL) từ ngày 1 tháng 1 năm 1990 đến ngày 28 tháng 5 năm 2024.
Kết quả trả về là một DataFrame chứa dữ liệu giá cổ phiếu của AAPL.
-->

---

### Download list of NASDAQ stock symbols

Go to the following URL to download a list of NASDAQ stock symbols (we will use _Mega_, _Large_ and _Medium_ for our Market Cap filter):

https://www.nasdaq.com/market-activity/stocks/screener

![NASDAQ stock screener](/nasdaq-stock-screener.png)

<!--
Giờ ta cần tải danh sách mã cổ phiếu của NASDAQ từ trang web của NASDAQ. Chúng ta sẽ sử dụng các bộ lọc _Mega_, _Large_ và _Medium_ cho _Market Cap_ của cổ phiếu.
Có thể thấy hiện tại có 1994 mã cổ phiếu có vốn thị trường lớn và trung bình.
-->

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

<!--
Sau khi tải danh sách mã cổ phiếu của NASDAQ, ta sẽ đọc lên danh sách mã cổ phiếu từ file CSV. Tiếp theo, ta sẽ lặp qua từng mã cổ phiếu để lấy dữ liệu giá cổ phiếu từ Yahoo Finance và lưu vào các file CSV tương ứng với mã cổ phiếu đó. Nếu có lỗi xảy ra, ta sẽ ghi lại mã cổ phiếu đó vào danh sách lỗi. Cuối cùng, ta sẽ cập nhật lại danh sách mã cổ phiếu sau khi loại bỏ các mã cổ phiếu lỗi.
-->

---
layout: image
image: "/crawling-data.png"
---

<!--
Lúc mà slide này được làm thì có tất cả 2014 mã cổ phiếu và 2 trong số đó bị lỗi nên còn lại 2012 mã cổ phiếu đã được lấy dữ liệu thành công.
-->

---
layout: image
image: "/data-crawled.png"
---

<!--
Các file CSV chứa dữ liệu giá cổ phiếu của các mã cổ phiếu đã được lưu vào thư mục csvdata.
-->

---

## Setting up the data

### Hypothesis formulation and in-sample testing

Starting by define numbers of days for holding the stock

```python
holding_days = 30
```

<!--
Giờ ta đã có các dữ liệu giá cổ phiếu của các mã cổ phiếu từ NASDAQ. Tiếp theo ta sẽ tiền xử lý dữ liệu để chuẩn bị cho việc xây dựng mô hình dự đoán giá cổ phiếu.
Ta sẽ bắt đầu bằng việc xác định số ngày giữ cổ phiếu.
-->

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

<!--
Hàm `long_returns` sẽ tính toán lợi nhuận của một lênh mua cổ phiếu trong một khoảng thời gian giữ cố định (số ngày). Hàm này giả định rằng người giao dịch không có kỹ năng và sẽ mua cổ phiếu với giá cao nhất của ngày trước và bán với giá thấp nhất của ngày khi đóng lệnh.
Thì giả sử người giao dịch bán với giá thấp nhất vào ngày 20/5, tức là người giao dịch đã mua với giá cao nhất vào ngày 20/4 (30 ngày trước đó) thì lợi nhuận sẽ là 0.1848%
-->

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

<!--
Tương tự hàm `short_returns` sẽ tính toán lợi nhuận của một lệnh bán cổ phiếu trong một khoảng thời gian giữ cố định (số ngày). Hàm này giả định rằng người giao dịch không có kỹ năng và sẽ bán cổ phiếu với giá thấp nhất của ngày 'numdays' trước đó và mua lại với giá cao nhất của ngày khi đóng lệnh.
-->

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

<!--
Từ đó ta sẽ dán nhãn dữ liệu dựa trên điều kiện của lợi nhuận mua và lợi nhuận bán. Nếu lợi nhuận mua lớn hơn 0.5%, ta cần mua. Nếu lợi nhuận bán lớn hơn 0.5%, ta cần bán. Ngưỡng 0.5% được sử dụng để tránh các giao dịch có thể dẫn đến lỗ do chi phí giao dịch. Ngưỡng này có thể được điều chỉnh tùy thuộc vào hiệu quả giao dịch của quỹ đầu tư.
-->

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

<!--
Hàm `moving_avg_data` sẽ tính toán các giá trị trung bình động cho dữ liệu cổ phiếu. Hàm này sẽ lặp qua từng tên biến và số ngày tương ứng để tính toán giá trị trung bình động. Cuối cùng, hàm sẽ trả về DataFrame với các cột giá trị trung bình động được thêm vào.
-->

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

<!--
Giờ ta sẽ tạo các bộ dữ liệu cho việc huấn luyện và kiểm tra mô hình. Ta sẽ chọn ngẫu nhiên các mã cổ phiếu từ danh sách mã cổ phiếu và lấy dữ liệu giá cổ phiếu từ các file CSV tương ứng. Sau đó, ta sẽ tính toán lợi nhuận mua và bán dựa trên dữ liệu trước đó. Tiếp theo, ta sẽ tính toán giá trị trung bình động cho dữ liệu cổ phiếu. Cuối cùng, ta sẽ lưu DataFrame vào file CSV.
-->

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

<!--
Ta sẽ đọc danh sách mã cổ phiếu từ file CSV. Sau đó, ta sẽ tạo các bộ dữ liệu huấn luyện và kiểm tra với 50 mã cổ phiếu ngẫu nhiên.
-->
