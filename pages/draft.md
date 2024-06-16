### Hypothesis formulation and in-sample testing

Perform scaling of moving averages by dividing the moving averages by the largest moving average value according to the hypothesis.

```python {all|2-5|7-10|12-15|all}
def scale_moving_averages(df, mavnames, mavdays):
    # Determine the name of the moving average column with the largest number of days
    maxmovavg = mavnames[mavdays.index(max(mavdays))]
    # Remove the name of the largest moving average column from the mavnames list
    mavnames.remove(maxmovavg)

    # Loop through the remaining moving average variable names for scaling
    for i in range(len(mavnames)):
        # Divide the values of the moving averages by the value of the largest moving average
        df[mavnames[i]] = df[mavnames[i]] / df[maxmovavg]

    # Set the value of the largest moving average to 1
    df.loc[:, maxmovavg] = 1
    # Remove the initial rows that do not have enough data for the largest moving average
    df.drop(df.index[:max(mavdays)], inplace=True)
    # Return the scaled DataFrame
    return df

result = scale_moving_averages(df, mavnames.copy(), mavdays)
result.tail()
```

---

### Hypothesis formulation and in-sample testing

The result

![Scaled moving averages](/scale-moving-averages.png)
