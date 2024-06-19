# Technical indicators - Moving averages

$\text{Well-known equation of asset prices: }$
$P_t = \mu t + \sigma \sqrt{t} + \epsilon$

- $\mu t$ (Trend Term): Represents the underlying trend in the asset price over time.
  - $\mu$: Constant indicating the average rate of return or drift.
  - Dominates as $t$ becomes large, showing overall trend direction.

- $\sigma \sqrt{t}$ (Volatility Term): Accounts for the variability in the asset price.
  - $\sigma$: Standard deviation of returns, indicating price variation.
  - Increases with $\sqrt{t}$, reflecting market volatility.

- $\epsilon$ (Noise Term): Represents random noise or shocks to the asset price, causes short-term deviations from the trend.

Objective of Moving Averages is to smooth data by removing noise ($\epsilon$) while retaining important information (trend $\mu t$ and volatility $\sigma \sqrt{t}$).

<!-- 
- Hạng tử mu*t (thành phần xu hướng) đại diện cho xu hướng giá tài sản theo thời gian
    - mu: Hằng số chỉ ra tỉ lệ lợi nhuận trung bình
    - Khi t lớn, thành phần này sẽ chiếm ưu thế và sẽ cho chúng ta cái nhìn tổng quát về xu hướng giá của tài sản
- Hạng tử sigma*sqrt(t) (thành phần biến động) đại diện cho sự biến đổi giá của tài sản
    - sigma: độ lệch chuẩn của lợi nhuận
- Epsilon (thành phần nhiễu): đại diện cho nhiễu ngẫu nhiên hoặc các cú sốc đến giá tài sản, gây ra các sai lệch ngắn hạn khỏi xu hướng tổng quát.
-->

---

# Technical indicators - Moving averages
## Candlestick chart with simple MA for BID stock (BIDV) - period 50 days

<img src='/ma-example.png' className='w-180 mx-auto'/>

