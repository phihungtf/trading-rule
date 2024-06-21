## Overview

### Topics Covered

- Simple Trading Rule
- Simple Classification Network
- Deep Convolutional Network
- Comparison of Model Performance

---

## Simple Trading Rule

### Concept

- Relies on trend persistence
- Uses moving averages to determine buy/sell signals

### Methodology

- Buy when short-term moving averages are above long-term moving averages
- Sell when the opposite is true

---

## Simple Trading Rule - Implementation

### Trading Rule Concept

- Buy Signal: 20-day moving average > 50-day moving average > 200-day moving average
- Sell Signal: 20-day moving average < 50-day moving average < 200-day moving average
- Note: Data is already scaled for the 200-day moving average

### Code Example

```python
def trading_rule_20_50_200(df):
    # Initialize RuleAction column with 'None'
    df['RuleAction'] = 'None'

    # Apply Buy rule
    df.loc[((df['mav20'] > df['mav50']) & (df['mav50'] > df['mav200'])), 'RuleAction'] = 'Buy'

    # Apply Sell rule
    df.loc[((df['mav20'] < df['mav50']) & (df['mav50'] < df['mav200'])), 'RuleAction'] = 'Sell'

    return df
```

---

## Simple Trading Rule - Implementation

### Applied to 50 random stock data

```python
csvfilename = 'train_50.csv'
data = pd.read_csv('sampledata/'+ csvfilename)
data['Action'].fillna('None', inplace=True)
data = trading_rule_20_50_200(data)
data.sample(5)
```

### Result

<img src='/simple-trading-rule-table.png' alt="simple-trading-rule-table" className="mx-auto"/>

---

## Simple Trading Rule - Accuracy Analysis

**Evaluating Prediction Accuracy**

```python
# Convert actions into binary matrices
ytest = np.array(pd.get_dummies(data.Action))  # Actual actions
predict_valid = np.array(pd.get_dummies(data.RuleAction))  # Predicted actions

# Evaluate prediction accuracy
df_accuracy = prediction_accuracy(ytest, predict_valid)

df_accuracy 
```

**Example of `ytest` và `predict_valid`**

| Action | One-Hot Encoding (ytest) | RuleAction | One-Hot Encoding (predict_valid) |
|--------|---------------------------|------------|----------------------------------|
| Buy    | [1, 0, 0]                 | Buy        | [1, 0, 0]                        |
| Buy    | [1, 0, 0]                 | Buy        | [1, 0, 0]                        |
| Sell   | [0, 0, 1]                 | None       | [0, 1, 0]                        |
| Buy    | [1, 0, 0]                 | Buy        | [1, 0, 0]                        |
| Sell   | [0, 0, 1]                 | None       | [0, 1, 0]                        |


---

## Simple Trading Rule - Accuracy Analysis


**Define `prediction_accuracy`**

```python
        def prediction_accuracy(ytest, predict_val):
            # Initialize a 3x3 accuracy matrix
            # Rows represent predictions, columns represent actual test values
            # Order: BUY, NONE, SELL
            accuracy_mat = np.zeros([3, 3], dtype=float)

            # Iterate through each column of ytest and predict_val
            for i in range(ytest.shape[1]):
                for j in range(predict_val.shape[1]):
                    # Calculate the sum of correct predictions for each pair (i, j)
                    accuracy_mat[i, j] = sum(predict_val[(predict_val[:, j] * ytest[:, i] > 0), j])

            # Calculate the total number of observations
            allobs = sum(map(sum, accuracy_mat))

            # Divide each element of the accuracy matrix by the total number of observations to get the percentage
            accuracy_mat = np.divide(accuracy_mat, allobs) * 100

            # Convert the accuracy matrix to a DataFrame with appropriate column and row labels
            accuracy_mat = pd.DataFrame(accuracy_mat, columns=['Buy', 'None', 'Sell'], index=['Buy', 'None', 'Sell'])

            # Return the accuracy matrix as a DataFrame
            return accuracy_mat

```
---

## Simple Trading Rule - Accuracy Analysis

<img src='/simple-trading-rule-heatmap.png' alt="simple-trading-rule-table" className="mx-auto" style="width: 50%;"/>

<div style="margin-right: 5%; margin-left: 5%;">

### Key Points

- #### Accuracy: ~50%
- #### Low accuracy and high misclassification rates
- #### False signals due to noise in signal levels
- #### Does not distinguish between temporary shifts in moving averages

</div>

---

## Simple Classification Network - Overview

### Topics Covered

- Simple Classification Network
- Diagnostic Chart
- Heatmap Analysis
- Performance Comparison

---

## Simple Classification Network - Background

### Background

- Inspired by Fisher's Iris dataset classification
- Uses a multilayer perceptron
- Three input layers, one hidden layer, and one output layer
- Fully connected with a sigmoid activation function

### Similarities to Our Problem

- Few characteristics (e.g., moving averages)
- Classification into three categories: Buy, Sell, Hold

<img src='/paper.png' className="absolute top-15 right-5 w-100 border-2"/>

<img src='/simple network.png' className='w-80'/>

---

## Simple Classification Network - Data Preparation

### Sample Data Distribution

```python
import pandas as pd
import seaborn as sns

csvfilename = 'train_50.csv'
data = pd.read_csv('sampledata/'+ csvfilename)
data = data[['mav5', 'mav10', 'mav20', 'mav30', 'mav50', 'mav100', 'Action']]
g = sns.pairplot(data, hue="Action", height=2.5)
g.savefig('figures/train_50_desc.png')
```

---

## Simple Classification Network - Diagnostic Chart

<img src='/diagnostic-chart.png' alt="Diagnostic Chart" className="mx-auto" style="height: 30%;"/>

---

## Simple Classification Network - Neural Network Implementation

### Code Example

<div style="display: flex;">

<div style="flex: 1; padding-right: 10px;">
    
```python
training_size = X_train.shape[1]
test_size = X_test.shape[1]
num_features = 6
num_labels = 3
num_hidden = 10

# Build network with TensorFlow

graph = tf.Graph()
with graph.as_default():
tf_train_set = tf.constant(X_train)
tf_train_labels = tf.constant(y_train)
tf_valid_set = tf.constant(X_test)

    print(tf_train_set)
    print(tf_train_labels)

    ## Note, since there is only 1 layer there are actually no hidden layers... but if there were
    ## there would be num_hidden
    weights_1 = tf.Variable(tf.random.truncated_normal([num_features, num_hidden]))
    weights_2 = tf.Variable(tf.random.truncated_normal([num_hidden, num_labels]))
```

</div>
</div>

---

## Simple Classification Network - Neural Network Implementation

### Code Example

<div style="display: flex;">

<div style="flex: 1; ">

```python
    ## tf.zeros Automaticaly adjusts rows to input data batch size
    bias_1 = tf.Variable(tf.zeros([num_hidden]))
    bias_2 = tf.Variable(tf.zeros([num_labels]))

    logits_1 = tf.matmul(tf_train_set , weights_1 ) + bias_1
    rel_1 = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(rel_1, weights_2) + bias_2

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_2, labels=tf_train_labels))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(.005).minimize(loss)

    ## Training prediction
    predict_train = tf.nn.softmax(logits_2)

    # Validation prediction
    logits_1_val = tf.matmul(tf_valid_set, weights_1) + bias_1
    rel_1_val    = tf.nn.relu(logits_1_val)
    logits_2_val = tf.matmul(rel_1_val, weights_2) + bias_2
    predict_valid = tf.nn.softmax(logits_2_val)
```

</div>

</div>

---

## Simple Classification Network - Training and Evaluation

### Code Example

```python
with graph.as_default():
    saver = tf.compat.v1.train.Saver()
num_steps = 10000
with tf.compat.v1.Session(graph = graph) as session:
    session.run(tf.compat.v1.global_variables_initializer())
    print(loss.eval())
    for step in range(num_steps):
        _,l, predictions = session.run([optimizer, loss, predict_train])

        if (step % 2000 == 0 or step == num_steps-1):
              #print(predictions[3:6])
              print('Loss at step %d: %f' % (step, l))
              print('Training accuracy: %.1f%%' % accuracy( predictions, y_train[:, :]))
              print('Validation accuracy: %.1f%%' % accuracy(predict_valid.eval(), y_test))
              predict_valid_arr = predict_valid.eval()
              saver.save(session,"simpleclass/bs.ckpt")
```

---

## Simple Classification Network - Training and Evaluation

### Results

<img src='/simple-trading-rule-train-validation.png' alt="Diagnostic Chart" className="mx-auto" style="height: 30%;"/>

---

## Simple Classification Network - Heatmap Analysis

### Heatmap Generation

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Predictions
predictions = model.predict(X_test)
prediction_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test.to_numpy(), axis=1)

# Accuracy matrix
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(true_classes, prediction_classes)

# Plot heatmap
plt.figure(figsize=(10,7))
sns.heatmap(matrix, annot=True, fmt="d", cmap='viridis')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('figures/heatmap.png')
plt.show()
```

---

## Simple Classification Network - Heatmap Analysis

### Heatmap

<img src='/simple-classification-network-heatmap.png' alt="Diagnostic Chart" className="mx-auto" style="height: 30%;"/>

---

## Simple Classification Network - Performance Summary

### Key Points

- **Accuracy Improvement**: From 50% to 70%
- **Misclassification**: Reduced significantly
- **New Benchmark**: Higher accuracy with reduced misclassification

---

## Simple Classification Network - Conclusion

### Key Takeaways

- Neural network outperforms simple trading rules
- Classification network achieves higher accuracy
- Effective in distinguishing between Buy, Sell, and Hold signals

### Future Work

- Explore deeper network architectures
- Evaluate with larger datasets
