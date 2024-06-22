# Overview

### Topics Covered
<ul className='text-3xl'>
<li>Simple Trading Rule</li>
<li>Simple Classification Network</li>
</ul>

<!-- Trong phần này ta sẽ cùng khám phá hai chủ đề chính: Quy tắc Giao dịch Đơn giản và Mạng Phân loại Đơn giản.  -->
---

# Simple Trading Rule

### Concept

<ul className='text-3xl'>
<li>Relies on trend persistence</li>
<li>Uses moving averages to determine buy/sell signals</li>
</ul>


<!-- Đầu tiên, chúng ta sẽ nói về Quy tắc Giao dịch Đơn giản. Quy tắc này dựa vào xu hướng của thị trường và sử dụng các đường trung bình động để xác định các tín hiệu mua và bán. -->

### Methodology

<ul className='text-3xl'>
<li>Buy when short-term moving averages are above long-term moving averages</li>
<li>Sell when the opposite is true</li>
</ul>

<!-- Phương pháp của chúng ta là mua khi các đường trung bình động ngắn hạn cao hơn đường trung bình động dài hạn và bán khi điều ngược lại xảy ra. -->

---

# Simple Trading Rule - Implementation

### Trading Rule Concept

- Buy Signal: 20-day moving average > 50-day moving average > 200-day moving average
- Sell Signal: 20-day moving average < 50-day moving average < 200-day moving average
- Note: Data is already scaled for the 200-day moving average

<!-- Quy tắc giao dịch cụ thể là mua khi đường trung bình động 20 ngày > 50 ngày > 200 ngày và bán khi 20 ngày < 50 ngày < 200 ngày. Dữ liệu đã được chuẩn hóa cho đường trung bình động 200 ngày. -->


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
<!-- Đây là ví dụ cài đặt của quy tắc giao dịch chúng ta vừa thảo luận. -->
---

# Simple Trading Rule - Implementation

### Applied to 50 random stock data

```python
csvfilename = 'train_50.csv'
data = pd.read_csv('sampledata/'+ csvfilename)
data['Action'].fillna('None', inplace=True)
data = trading_rule_20_50_200(data)
data.sample(5)
```
<!-- Chúng ta đã áp dụng quy tắc này trên 50 dữ liệu chứng khoán ngẫu nhiên. Hình ảnh bên dưới hiển thị kết quả của việc áp dụng quy tắc giao dịch này. -->

### Result

<img src='/simple-trading-rule-table.png' alt="simple-trading-rule-table" className="mx-auto"/>

---

# Simple Trading Rule - Accuracy Analysis

**Evaluating Prediction Accuracy**

```python
# Convert actions into binary matrices
ytest = np.array(pd.get_dummies(data.Action))  # Actual actions
predict_valid = np.array(pd.get_dummies(data.RuleAction))  # Predicted actions

# Evaluate prediction accuracy
df_accuracy = prediction_accuracy(ytest, predict_valid)

df_accuracy 
```
<!-- Bây giờ chúng ta sẽ đánh giá độ chính xác của dự đoán bằng cách chuyển đổi các hành động thành ma trận nhị phân và sử dụng hàm prediction_accuracy. -->

**Example of `ytest` và `predict_valid`**

| Action | One-Hot Encoding (ytest) | RuleAction | One-Hot Encoding (predict_valid) |
|--------|---------------------------|------------|----------------------------------|
| Buy    | [1, 0, 0]                 | Buy        | [1, 0, 0]                        |
| Buy    | [1, 0, 0]                 | Buy        | [1, 0, 0]                        |
| Sell   | [0, 0, 1]                 | None       | [0, 1, 0]                        |
| Buy    | [1, 0, 0]                 | Buy        | [1, 0, 0]                        |
| Sell   | [0, 0, 1]                 | None       | [0, 1, 0]                        |


---

# Simple Trading Rule - Accuracy Analysis


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

# Simple Trading Rule - Accuracy Analysis

<img src='/simple-trading-rule-heatmap.png' alt="simple-trading-rule-table" className="mx-auto" style="width: 50%;"/>

<div style="margin-right: 5%; margin-left: 5%;">

### Key Points

- #### Accuracy: ~50%
- #### Low accuracy and high misclassification rates
- #### False signals due to noise in signal levels
- #### Does not distinguish between temporary shifts in moving averages

</div>

<!-- Hình ảnh dưới đây là ma trận nhiệt hiển thị độ chính xác của quy tắc giao dịch. Chúng ta có thể thấy độ chính xác chỉ khoảng 50%, với tỷ lệ sai sót cao
+ có thể là do các tín hiệu nhiễu
+ không phân biệt được giữa các thay đổi tạm thời trong các đường trung bình động. 
(Ví dụ, một sự kiện kinh tế hoặc tin tức ngắn hạn có thể khiến giá cổ phiếu tăng hoặc giảm đột ngột, nhưng sự thay đổi này chỉ là tạm thời và không duy trì trong thời gian dài. Trong trường hợp này, đường trung bình động 20 ngày có thể vượt lên hoặc xuống qua đường trung bình động 50 ngày và 200 ngày, tạo ra tín hiệu mua hoặc bán theo quy tắc giao dịch. Tuy nhiên, vì đây chỉ là sự thay đổi ngắn hạn, tín hiệu này có thể không chính xác và dẫn đến giao dịch thua lỗ khi xu hướng thực sự của thị trường không thay đổi. )
 > Điều này dẫn đến các tín hiệu giao dịch sai, làm giảm hiệu quả của quy tắc giao dịch và tăng tỷ lệ sai sót.
-->
---

# Simple Classification Network - Background

### Background

- Inspired by Fisher's Iris dataset classification
- Uses a multilayer perceptron
- Three input layers, one hidden layer, and one output layer
- Fully connected with a sigmoid activation function

### Similarities to Our Problem

- Few characteristics (e.g., moving averages)
- Classification into three categories: Buy, Sell, Hold

<img src='/paper.png' className="absolute top-25 right-5 w-100 border-2"/>

<img src='/simple network.png' className='w-80'/>

<!--
Tiếp theo, tao sẽ giới thiệu về Mạng Phân Loại Đơn Giản.

Trước hết, mạng này được lấy cảm hứng từ bài toán phân loại bộ dữ liệu Iris của Fisher, một bài toán kinh điển trong học máy. 
Bài toán phân loại bộ dữ liệu Iris của Fisher nhằm xác định loài hoa Iris (3 loại) dựa trên các đặc trưng: chiều dài và chiều rộng củ cánh hoa, màu săc.

Tương tự với bài toán trading của chúng ta cũng là một bài toán phân loại, trong đó chúng ta dự đoán hành động mua, bán hoặc giữ dựa trên các đặc trưng là các đường trung bình động của giá cổ phiếu.

Perceptron đa lớp (Multilayer Perceptron - MLP) là một loại mạng nơ-ron nhân tạo, gồm ít nhất ba lớp nơ-ron: một lớp đầu vào, một hoặc nhiều lớp ẩn, và một lớp đầu ra. Các nơ-ron trong mỗi lớp được kết nối hoàn toàn với các nơ-ron ở lớp kế tiếp.

Ví dụ cấu trúc MLP cho bài toán Iris:
    Lớp đầu vào: Gồm bốn nơ-ron, mỗi nơ-ron tương ứng với một trong các đặc trưng của hoa Iris.
    Lớp ẩn: Một hoặc nhiều lớp ẩn, mỗi lớp có một số lượng nơ-ron nhất định. Lớp ẩn giúp mạng nơ-ron học được các mối quan hệ phức tạp giữa các đặc điểm của loài hoa để phân loại chính xác loài hoa.
    Lớp đầu ra: Gồm ba nơ-ron, mỗi nơ-ron tương ứng với một trong ba loại hoa Iris 

Hàm kích hoạt sigmoid: Trong mạng MLP của bạn, các giá trị đầu ra từ mỗi nơ-ron trong lớp ẩn được chuyển qua hàm sigmoid để biến đổi thành giá trị trong khoảng từ 0 đến 1. Được sử dụng trong lớp ẩn để giúp mô hình học được các mối quan hệ phi tuyến giữa các đặc trưng và nhãn. 
-->

---

# Simple Classification Network - Data Preparation

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
<!-- - Đầu tiên, chúng ta sẽ xem xét phân bố dữ liệu của mẫu dữ liệu 50 cổ phiếu.
- Sử dụng code Python để đọc dữ liệu từ tệp CSV và chọn các cột cần thiết.
- Sau đó, tạo biểu đồ cặp đôi để trực quan hóa phân bố của các đặc trưng với nhau, phân theo hành động (Action). -->
---

# Simple Classification Network - Diagnostic Chart

<img src='/diagnostic-chart.png' alt="Diagnostic Chart" className="mx-auto w-150"/>

<!-- 
-  Biểu đồ cặp cho thấy sự phân bố của các điểm dữ liệu trong không gian đặc trưng, tức là không gian mà các đặc trưng (các trung bình động) chiếm giữ.
- Trên biểu đồ thể hiện các điểm dữ liệu xếp thành một mẫu hình rõ ràng, điều đó chỉ ra rằng có mối tương quan chặt chẽ giữa các đặc trưng.
- Mức độ tương quan cao này có thể là một dấu hiệu tốt để sử dụng các đặc trưng này trong mô hình phân loại, vì chúng cung cấp thông tin hữu ích về xu hướng và quan hệ giữa các biến.
-->
---


# Simple Classification Network - Neural Network Implementation

#### Handling Missing Values and Data Preparation
```python
data['Action'].fillna('None', inplace=True)  # Replace NaN values in 'Action' column with 'None'

cols = data.columns  # Get all column names
features = cols[0:6]  # Select the first 6 columns as features
labels = cols[6]  # Select the 7th column as labels

indices = data.index.tolist()  # Get the indices of the data
indices = np.array(indices)  # Convert indices to numpy array
np.random.shuffle(indices)  # Shuffle the indices

# Reindex the data based on the shuffled indices
X = data.reindex(indices)[features]
y = data.reindex(indices)[labels]

# Convert categorical labels to binary matrix
y = pd.get_dummies(y)
```
<!-- 
Xáo trộn dữ liệu và tách các đặc trưng và nhãn cho quá trình huấn luyện:
- indices lấy danh sách các chỉ mục của data.
- chuyển indices thành mảng numpy và xáo trộn ngẫu nhiên.
- X là dữ liệu đặc trưng sau khi xáo trộn. y là nhãn của dữ liệu sau khi xáo trộn.

Xáo trộn dữ liệu giúp đảm bảo rằng các mẫu được phân phối ngẫu nhiên, tránh hiện tượng overfitting: ví dụ: tất cả các mẫu của một lớp cụ thể nằm liền kề nhau), mô hình có thể học được sự phụ thuộc vào thứ tự này thay vì học các đặc trưng thực sự của dữ liệu, dẫn đến hiện tượng overfitting.
 -->

---

# Simple Classification Network - Neural Network Implementation

#### Splitting Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # Split data into training and testing sets

# Convert data to numpy arrays and change dtype to float32
X_train = np.array(X_train).astype(np.float32)
X_test  = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test).astype(np.float32)

# Display shapes of the datasets
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```
```python
((185321, 6), (79424, 6), (185321, 3), (79424, 3))
```

**Explanation:**
  - `X_train.shape` and `X_test.shape`: Shapes of training and testing feature sets.
  - `y_train.shape` and `y_test.shape`: Shapes of training and testing label sets.

<!-- 
- Trước hết, chúng ta cần chia dữ liệu thành các tập huấn luyện và kiểm tra.
- Sử dụng hàm `train_test_split` từ thư viện `sklearn` để chia dữ liệu với tỷ lệ 70% cho huấn luyện và 30% cho kiểm tra.
- Chuyển đổi dữ liệu thành các mảng numpy và thay đổi kiểu dữ liệu thành float32 để phù hợp với mô hình TensorFlow.

Giải thích shape:
- `X_train.shape`: 185,321 mẫu trong tập huấn luyện, mỗi mẫu có 6 đặc trưng.
- `X_test.shape`: 79,424 mẫu trong tập kiểm tra, mỗi mẫu có 6 đặc trưng.
- `y_train.shape`: 185,321 mẫu trong tập huấn luyện, mỗi mẫu có một nhãn mã hóa one-hot của 3 lớp (Mua, Giữ, Bán).
- `y_test.shape`: 79,424 mẫu trong tập kiểm tra, mỗi mẫu có một nhãn mã hóa one-hot của 3 lớp (Mua, Giữ, Bán).
 -->

---



# Simple Classification Network - Neural Network Implementation

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

    weights_1 = tf.Variable(tf.random.truncated_normal([num_features, num_hidden]))
    weights_2 = tf.Variable(tf.random.truncated_normal([num_hidden, num_labels]))
```

</div>
</div>

<!-- 
Định nghĩa biến và chuẩn bị dữ liệu:
-   training_size và test_size: Xác định số lượng đặc trưng (features) trong dữ liệu huấn luyện và kiểm tra. Giá trị này là 6 (mav5, mav10, mav20, mav30, mav50, mav100).
-   num_features: Số lượng đặc trưng trong dữ liệu (6).
-   num_labels: Số lượng nhãn (labels) đầu ra. Trong trường hợp này, chúng ta có 3 nhãn (Buy, Sell, Hold).
-   num_hidden: Số lượng neuron trong lớp ẩn của mạng neural (10).

Sau dó là các bước xây dựng các lớp của mạng neural

weights_1

-	Kích thước: [num_features, num_hidden]
-	Cụ thể: Khi dữ liệu đầu vào (tf_train_set) đi qua lớp này, nó sẽ được nhân với ma trận trọng số weights_1. Điều này tạo ra các giá trị trọng số mới cho mỗi đặc trưng đầu vào dựa trên mức độ quan trọng của chúng.

weights_2

-	Kích thước: [num_hidden, num_labels]
-	Cụ thể: Sau khi các giá trị từ lớp ẩn được tính toán, chúng sẽ được nhân với ma trận trọng số weights_2 để tạo ra các giá trị cuối cùng ở lớp đầu ra.

Trọng số (weights_1, weights_2): Quyết định mức độ quan trọng của mỗi đặc trưng đầu vào và các giá trị từ lớp ẩn. Chúng là các tham số học được trong quá trình huấn luyện để tối ưu hóa hàm mất mát.

 -->
---

# Simple Classification Network - Neural Network Implementation

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

<!-- 
1. bias_1
-	Kích thước: [num_hidden]
-	Tác dụng: Đây là vector bias được thêm vào các giá trị từ lớp đầu vào sau khi chúng được nhân với weights_1.
-	Cụ thể: Tham số bias giúp điều chỉnh đầu ra của mỗi neuron trong lớp ẩn, đảm bảo rằng mạng neural có thể dịch chuyển hàm kích hoạt mà không phụ thuộc hoàn toàn vào giá trị đầu vào.
2. bias_2
-	Kích thước: [num_labels]
-	Tác dụng: Đây là vector bias được thêm vào các giá trị từ lớp ẩn sau khi chúng được nhân với weights_2.
-	Cụ thể: Tham số bias giúp điều chỉnh đầu ra của mỗi neuron trong lớp đầu ra, tương tự như bias_1, đảm bảo rằng mạng có thể điều chỉnh kết quả đầu ra độc lập với đầu vào từ lớp trước.

**Tham số bias (bias_1, bias_2): Giúp dịch chuyển đầu ra của các neuron, cho phép mạng neural điều chỉnh kết quả mà không phụ thuộc hoàn toàn vào giá trị đầu vào. Điều này làm cho mạng có thể học được nhiều dạng mẫu dữ liệu hơn.**


logits_1
-	Tác dụng: logits_1 là giá trị đầu ra của lớp đầu tiên sau khi thực hiện phép **nhân ma trận giữa dữ liệu đầu vào (tf_train_set) và trọng số (weights_1), sau đó cộng thêm tham số bias (bias_1).**
-	Cụ thể: logits_1 **đại diện cho sự kết hợp tuyến tính của các đặc trưng đầu vào trước khi áp dụng hàm kích hoạt**. Nó là bước tính toán đầu tiên trong mạng neural và sẽ là **đầu vào cho lớp kích hoạt tiếp theo (rel_1)**.

rel_1
-	Tác dụng: rel_1 là giá trị đầu ra của lớp đầu tiên sau khi áp dụng hàm kích hoạt ReLU (Rectified Linear Unit) lên logits_1.
-	Cụ thể: **Hàm ReLU chuyển đổi các giá trị âm trong logits_1 thành 0, trong khi giữ nguyên các giá trị dương. Điều này giúp mạng neural có tính không tuyến tính, cho phép nó học được các mối quan hệ phức tạp hơn giữa đầu vào và đầu ra.**

logits_2
-	Tác dụng: logits_2 là giá trị đầu ra của lớp thứ hai (lớp đầu ra cuối cùng) sau khi thực hiện phép nhân ma trận giữa rel_1 (đầu ra của lớp ẩn) và trọng số (weights_2), sau đó cộng thêm tham số bias (bias_2).
-	Cụ thể: logits_2 đại diện cho sự kết hợp tuyến tính cuối cùng trước khi áp dụng hàm kích hoạt ở đầu ra (ví dụ: hàm softmax). Nó là bước tính toán cuối cùng trước khi tính toán hàm mất mát và thực hiện tối ưu hóa.


loss
-	Tác dụng: Tính toán giá trị mất mát (loss) cho quá trình huấn luyện mạng neural.
-	Giải thích cụ thể:
-	tf.nn.softmax_cross_entropy_with_logits:
-	Hàm này tính toán giá trị mất mát sử dụng phương pháp cross-entropy (entropy chéo) giữa các logits và các nhãn thực tế.
-	Vai trò:
	**Giá trị mất mát (loss) là thước đo chính để đánh giá mức độ chênh lệch giữa dự đoán của mô hình và nhãn thực tế.
	Mục tiêu của quá trình huấn luyện là giảm thiểu giá trị mất mát này.**

optimizer
-	Đối tượng Optimizer sử dụng thuật toán Gradient Descent để cập nhật các trọng số và bias theo hướng giảm dần giá trị mất mát.

Training Prediction
- Hàm softmax biến đổi các giá trị logits thành các xác suất, với tổng xác suất của tất cả các lớp bằng 1. Kết quả này là xác suất dự đoán của mô hình cho từng lớp trong quá trình huấn luyện.

Validation: tương tự …
 -->
---

# Simple Classification Network - Training and Evaluation

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
<!-- 
- Đoạn mã này minh họa cách chúng ta huấn luyện và đánh giá mạng nơ-ron đơn giản.
- Đầu tiên, khởi tạo một phiên làm việc với đồ thị TensorFlow đã định nghĩa.
- Chạy vòng lặp huấn luyện trong 10,000 bước, tại mỗi bước tính toán mất mát và đưa ra dự đoán.
- In ra mất mát và độ chính xác huấn luyện mỗi 2,000 bước hoặc khi hoàn thành.
- Đánh giá độ chính xác trên tập kiểm tra và lưu mô hình.
### Hàm `accuracy`

- Hàm `accuracy` được sử dụng để tính toán độ chính xác của mô hình.
- Cụ thể, nó so sánh dự đoán của mô hình với nhãn thực tế và tính phần trăm chính xác.
- Hàm này chuyển đổi dự đoán và nhãn thành dạng chỉ số của lớp dự đoán cao nhất và so sánh chúng.

 -->
---

# Simple Classification Network - Training and Evaluation

### Results

<img src='/simple-trading-rule-train-validation.png' alt="Diagnostic Chart" className="mx-auto w-70"/>

<!-- 
- Hình này hiển thị kết quả của quá trình huấn luyện và kiểm tra mô hình.
- Chúng ta có thể thấy độ chính xác huấn luyện và kiểm tra qua các bước huấn luyện.
- Mô hình dần dần cải thiện độ chính xác và giảm mất mát qua thời gian.
- thêm: validation accuracy cao và gần với training accuracy, điều đó cho thấy mô hình có khả năng tổng quát hóa tốt và hoạt động tốt trên dữ liệu mới
 -->
---

# Simple Classification Network - Heatmap Analysis

### Heatmap Generation

```python
# save the results
hf = h5py.File('h5files/simpleclass_train_50.h5', 'w')
hf.create_dataset('predict_valid', data=predict_valid_arr)
hf.create_dataset('y_test', data = y_test)
hf.close()
```

```python
hf = h5py.File('h5files/simpleclass_train_50.h5', 'r')
predict_val = hf['predict_valid'][:]
ytest = hf['y_test'][:]
hf.close()
x = np.argmax(predict_val, axis = 1)
predict_valid = np.zeros(predict_val.shape)
predict_valid[x == 0,0] = 1
predict_valid[x == 1,1] = 1
predict_valid[x == 2,2] = 1
df = prediction_accuracy(ytest, predict_valid)
ax = sns.heatmap(df, annot=True, fmt="g", cmap='viridis')
ax.xaxis.set_ticks_position('top')
ax.figure.savefig('figures/simpleclass_50_50.png')
```

<!-- 
- Tiếp theo, chúng ta sẽ lưu và đọc lại kết quả dự đoán từ mô hình đã huấn luyện.
- Sử dụng thư viện h5py để lưu kết quả vào tệp .h5 và sau đó đọc lại.
- Chuyển đổi dự đoán thành dạng nhị phân và tính toán độ chính xác dự đoán.
- Cuối cùng, tạo heatmap để trực quan hóa độ chính xác của từng lớp dự đoán.
 -->
---

# Simple Classification Network - Heatmap Analysis

### Heatmap

<img src='/simple-classification-network-heatmap.png' alt="Diagnostic Chart" className="mx-auto" style="height: 30%;"/>

---

# Simple Classification Network - Performance Summary

### Key Points

- **Accuracy Improvement**: From 50% to 71%
- **Misclassification**: Reduced significantly
- **New Benchmark**: Higher accuracy with reduced misclassification

<!-- 
- Đầu tiên, độ chính xác đã được cải thiện từ 50% lên 71%, một bước nhảy đáng kể.
- Thứ hai, tỷ lệ phân loại sai đã giảm đáng kể, cho thấy mô hình này hiệu quả hơn nhiều so với quy tắc giao dịch đơn giản.
- Đây là một tiêu chuẩn mới với độ chính xác cao hơn và tỷ lệ sai sót giảm.
- Những cải tiến này chỉ ra rằng mạng nơ-ron phân loại có khả năng phân biệt tốt hơn giữa các tín hiệu mua, bán và giữ, mang lại kết quả chính xác và tin cậy hơn.
 -->
