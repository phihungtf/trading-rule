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

---

# Technical indicators - Moving averages
## Candlestick chart with simple MA for BID stock (BIDV) - period 50 days

<img src='/ma-example.png' className='w-180 mx-auto'/>