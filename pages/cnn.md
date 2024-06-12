<h1 className="text-lg font-semibold mb-2">Human Decision Process in Trading</h1>
<div className="flex items-center justify-between">
        <div className="flex-2">
            <ul className="list-disc ml-5 mt-10 text-xl">
            <li>Filter noise by observing multiple data points</li>
            <li>Confidence builds with consistent price trends</li>
            <li>Compare new data to historical patterns</li>
            <li>Act if patterns match confidently</li>
            <li>CNNs automate this decision-making</li>
            <li>Apply across multiple assets</li>
            </ul>
        </div>
        <div className="flex-1 text-center">
            <img src='/decision.png' alt="Decision Process" className="mx-auto"/>
        </div>
    </div>

---

<h1 className="text-lg font-semibold mb-2">Modeling Investment Logic</h1>
<ul className="list-disc ml-5 mt-10 text-xl">
        <li>Optimal observation period before trading</li>
        <li>Use moving-average data over 12 days</li>
        <li>Take most frequent action</li>
        <li>Hold if multiple crossovers or confusion</li>
        <li>If 8 out of 12 actions are to buy, consider buying</li>
        <li>Identify patterns to validate actions</li>
      </ul>

---

<h1 className="text-lg font-semibold mb-2">Selecting the Network Architecture</h1>
<div className="flex mt-10">
      <div className="flex-2 pr-4">
        <p className="mb-4">Selecting the appropriate architecture depends on the input data:</p>
        <ul className="list-disc ml-5">
          <li>Input defined as a tensor to capture temporal features</li>
          <li>Utilize multiple data points to capture volatility</li>
          <li>Convolutional layer size depends on data analysis depth</li>
          <li>Use bigger convolutional windows for capturing moving averages</li>
        </ul>
        <p className="mt-4">Architecture selection is an art rather than a science:</p>
        <p>Heuristics can guide rather than dictate the process</p>
      </div>
      <div className="flex-1">
        <img src="/decision-making.png" alt="Decision making" className="w-full h-auto" />
      </div>
    </div>

---

<h1 className="text-lg font-semibold mb-2">CNN Architecture</h1>
<ul className="list-disc ml-5 mt-10">
  <li>An input layer: 12 x 3 tensor representing 12 observations of 3 features</li>
  <li>A convolutional layer: Convolution to a space of a 6 x 6 tensor to capture patterns</li>
  <li>A pooling layer to flatten the data</li>
  <li>An output layer with 3 classes</li>
</ul>

---

<div className="flex flex-col space-y-2">
      <h1 className="text-lg font-semibold">Setting up the data in the correct format</h1>
      <h2 className="text-lg font-semibold mt-4">Explanation of Data Segmentation</h2>
      <ul className="list-disc list-inside mt-4">
        <li>Data is segmented into smaller segments.</li>
        <li>Each segment typically contains 12 observations.</li>
        <li>Overlapping windows capture patterns across data points.</li>
        <li>Segmented data is converted to NumPy arrays for CNN input.</li>
        <li>Arrays are saved in h5 files for efficient retrieval.</li>
        <li>Utility functions aid TensorFlow code implementation.</li>
      </ul>
</div>

---

<div className="flex flex-col space-y-2 mb-10">
      <h1 className="text-lg font-semibold">Setting up the data in the correct format</h1>
      <h2 className="text-lg font-semibold mt-4">Split the data into windows of the required time horizon</h2>
</div>


```python
def windows(data, size):
    # Hàm này tạo ra các cửa sổ (windows) từ dữ liệu với kích thước được định nghĩa bởi 'size'.
    start = 0  # Khởi tạo biến bắt đầu tại vị trí 0
    while start < data.count():  # Tiếp tục tạo cửa sổ cho đến khi vị trí bắt đầu vượt quá số lượng phần tử trong dữ liệu
        yield int(start), int(start + size)  # Trả về một cặp giá trị (start, start + size) đại diện cho cửa sổ dữ liệu hiện tại
        start += (size / 2)  # Di chuyển vị trí bắt đầu thêm nửa kích thước cửa sổ để tạo ra cửa sổ mới (các cửa sổ sẽ chồng lấn lên nhau)
```

---

<div className="flex flex-col space-y-2 mb-10">
      <h1 className="text-lg font-semibold">Setting up the data in the correct format</h1>
      <h2 className="text-lg font-semibold mt-4">Segment the data into a signal and its corresponding label</h2>
</div>


```python
def segment_signal(data,window_size = 12):
    segments = np.empty((0,window_size,6))
    labels = np.empty((0))
    for (start, end) in windows(data['Date'], window_size):
        x = data["mav5"][start:end]
        y = data["mav10"][start:end]
        z = data["mav20"][start:end]
        a = data["mav30"][start:end]
        b = data["mav50"][start:end]
        c = data["mav100"][start:end]
        if(len(data['Date'][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z,a,b,c])])
            unique_labels, counts = np.unique(data["Action"][start:end], return_counts=True)
            most_frequent_label = unique_labels[np.argmax(counts)]
            labels = np.append(labels, most_frequent_label)
    return segments, labels
```

---

<div className="flex flex-col space-y-2 mb-10">
      <h1 className="text-lg font-semibold">Setting up the data in the correct format</h1>
      <h2 className="text-lg font-semibold mt-4">Create batches from these segments that can be used to train our model</h2>
</div>


```python
def get_batches(X, y, batch_size=100):
    """ Trả về một generator cho các batch """

    # Tính số lượng batch có thể tạo ra
    n_batches = len(X) // batch_size

    # Cắt bớt dữ liệu X và y để phù hợp với số lượng batch
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Lặp qua các batch và trả về từng batch
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]

# Mạng nơ-ron được huấn luyện theo từng batch để tránh tình trạng tràn bộ nhớ.
```

---

<div className="flex flex-col space-y-1 mb-2 mt-0">
      <h1 className="text-lg font-semibold">Setting up the data in the correct format</h1>
      <h2 className="text-lg font-semibold">Create the training and test data in a format that can be used by TensorFlow</h2>
</div>

````md magic-move
```python
def create_tensorflow_train_data(csvfilename):
    df = pd.read_csv('sampledata/'+ csvfilename)
    df = df[['Date','symbolid','buyret','sellret','Action','mav5', 'mav10','mav20','mav30','mav50','mav100']]
    df['Action'].fillna('None', inplace=True)
    symbols = df.symbolid.unique()
    segments, labels = segment_signal(df[df.symbolid == symbols[0]])
    df = df[df.symbolid != symbols[0]]
    symbols = symbols[1:]
    for i in range(0,len(symbols)):
        x, a = segment_signal(df[df.symbolid == symbols[i]])
        segments = np.concatenate((segments, x), axis = 0)
        labels = np.concatenate((labels, a), axis = 0)
        df = df[df.symbolid != symbols[i]]
        print(str(round(i/len(symbols)*100,2)) + ' percent done')
    list_ch_train = pd.get_dummies(labels)
    list_ch_train = np.asarray(list_ch_train.columns)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    X_tr, X_vld, lab_tr, lab_vld = train_test_split(segments, labels, stratify = labels, random_state = 123)

    return X_tr, X_vld, lab_tr, lab_vld, list_ch_train
```
```python
def create_tensorflow_test_data(csvfilename):
    df = pd.read_csv('sampledata/'+ csvfilename)
    df = df[['Date','symbolid','buyret','sellret','Action','mav5', 'mav10','mav20','mav30','mav50','mav100']]
    df['Action'].fillna('None', inplace=True)
    list_ch_test = df.Action.unique()
    symbols = df.symbolid.unique()
    segments, labels = segment_signal(df[df.symbolid == symbols[0]])
    df = df[df.symbolid != symbols[0]]
    symbols = symbols[1:]
    for i in range(0,len(symbols)):
        x, a = segment_signal(df[df.symbolid == symbols[i]])
        segments = np.concatenate((segments, x), axis = 0)
        labels = np.concatenate((labels, a), axis = 0)
        df = df[df.symbolid != symbols[i]]
        print(str(round(i/len(symbols)*100,2)) + ' percent done')
    list_ch_test = pd.get_dummies(labels)
    list_ch_test = np.asarray(list_ch_test.columns)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    X_test = segments
    y_test = labels
    return X_test, y_test, list_ch_test
```
````

---

<div className="flex flex-col space-y-1 mb-2 mt-0">
      <h1 className="text-lg font-semibold">Setting up the data in the correct format</h1>
      <h2 className="text-lg font-semibold">The training and test data saved as h5 files</h2>
</div>

````md magic-move
```python
def savedataset(csvfilename, h5filename):
	dt = h5py.special_dtype(vlen=str)
	hf = h5py.File('h5files/'+ h5filename, 'w')
	for i in range(0,len(csvfilename)):
		csvfile = csvfilename[i]
		if(csvfile[:4] != 'test'):
			X_tr, X_vld, lab_tr, lab_vld, list_ch_train = create_tensorflow_train_data(csvfile)
			hf.create_dataset('X_tr',data=X_tr)
			hf.create_dataset('X_vld',data=X_vld)
			hf.create_dataset('lab_tr',data=lab_tr)
			hf.create_dataset('lab_vld',data=lab_vld)
			hf.create_dataset('list_ch_train',data=list_ch_train, dtype = dt)
			del(X_tr, X_vld, lab_tr, lab_vld, list_ch_train)
		else:
			X_test, y_test, list_ch_test = create_tensorflow_test_data(csvfile)
			hf.create_dataset('X_test', data = X_test)
			hf.create_dataset('y_test', data = y_test)
			hf.create_dataset('list_ch_test', data = list_ch_test, dtype = dt)
			del(X_test, y_test, list_ch_test)
	hf.close()
savedataset(['test_50.csv', 'train_50.csv'], 'hdf_50.h5')
```
```python
def get_tf_train_data(h5filename):

    hf = h5py.File('h5files/' + h5filename, 'r')
    X_tr = hf['X_tr'][:]
    X_vld = hf['X_vld'][:]
    lab_tr = hf['lab_tr'][:]
    lab_vld = hf['lab_vld'][:]
    list_ch_train = hf['list_ch_train'][:]
    hf.close()
    return X_tr, X_vld, lab_tr, lab_vld, list_ch_train

def get_tf_test_data(h5filename):

    hf = h5py.File('h5files/' + h5filename, 'r')
    X_test = hf['X_test'][:]
    y_test = hf['y_test'][:]
    list_ch_test = hf['list_ch_test'][:]

    return X_test, y_test, list_ch_test
```
````

---

<div className="flex flex-col space-y-2 mb-10">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Read the train data</h2>
</div>


```python
traindtfile = 'hdf_50.h5'
testdtfile = 'hdf_50.h5'
losssavefig = 'cnn_train_50_loss.png'
accsavefig = 'cnn_train_50_accuracy.png'
resultsave = 'cnn_train_50.h5'
chkpointdir = 'cnn-50/'

X_tr, X_vld, y_tr, y_vld, list_ch_train = get_tf_train_data(traindtfile)
```

---

<div className="flex flex-col space-y-2 mb-10">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Define hyperparameters</h2>
</div>


```python
batch_size = 600       # Batch size
seq_len = 12          # Number of steps
learning_rate = 0.0001
epochs = 1000
```

---

<div className="flex flex-col space-y-2 mb-10">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Set up TensorFlow graph</h2>
</div>

````md magic-move
```python
n_classes = 3 # buy sell and nothing
n_channels = 6 # moving averages

graph = tf.Graph()
```
```python
# Construct placeholders
with graph.as_default():
    inputs_ = tf.compat.v1.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.compat.v1.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.compat.v1.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.compat.v1.placeholder(tf.float32, name = 'learning_rate')
```
````

---

<div className="flex flex-col space-y-2 mb-10">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Define the CNN</h2>
</div>

```python
with graph.as_default():
    # (batch, 12, 3) --> (batch, 6, 6)
    conv1 = tf.keras.layers.Conv1D(filters=6, kernel_size=2, strides=1,
                                   padding='same', activation=tf.nn.relu)(inputs_)
    max_pool_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(conv1)

with graph.as_default():
    # Flatten and add dropout
    flat = tf.compat.v1.reshape(max_pool_1, (-1, 6*6))
    flat = tf.compat.v1.nn.dropout(flat, keep_prob=keep_prob_)
    # Predictions
    logits = tf.keras.layers.Dense(n_classes)(flat)
    soft = tf.compat.v1.argmax(logits,1)
    pred = tf.compat.v1.nn.softmax(logits,1)
    # Cost function and optimizer
    cost = tf.compat.v1.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_).minimize(cost)
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
```

---

<div className="flex flex-col space-y-1 mb-4">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">At each checkpoint, the efficacy of the network is saved</h2>
</div>

```python
if (os.path.exists(chkpointdir) == False):
    os.makedirs(chkpointdir)

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.compat.v1.train.Saver()
```

---

<div className="flex flex-col space-y-1 mb-4">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Now, we can run our training on the convolutional network we defined earlier.</h2>
</div>
````md magic-move
```python
with tf.compat.v1.Session(graph=graph) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    iteration = 1

    # Loop over epochs
    for e in range(epochs):

        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):

            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}

            # Loss
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
```
```python
# Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))

            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):
                val_acc_ = []
                val_loss_ = []

                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}

                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
```
```python
# Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))

            # Iterate
            iteration += 1


    saver.save(sess,chkpointdir + "bs.ckpt")
```
````

---

<div className="flex flex-col space-y-1 mb-4">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Plot the trajectory of the training accuracy and validation accuracy</h2>
</div>

```python
t = np.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 == 0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('figures/'+losssavefig)

# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, np.array(train_acc), 'r-', t[t % 10 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('figures/'+accsavefig)

del(X_tr, X_vld, y_tr, y_vld, list_ch_train)
```

---

<div className="flex flex-col space-y-1 mb-4">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Plot the trajectory of the training accuracy and validation accuracy</h2>
</div>

<div class="flex space-x-2 justify-between">
 <img src='/loss.png' alt="Loss" className="w-1/2"/>
 <img src='/acc.png' className="w-1/2" alt="Accuracy"/>
</div>

---

<div className="flex flex-col space-y-1 mb-4">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Save the predictions</h2>
</div>

````md magic-move
```python
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint(chkpointdir))

    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}

        batch_acc = sess.run(accuracy, feed_dict=feed)
        test_acc.append(batch_acc)
        prob = sess.run(pred, feed_dict=feed)
        probs.append(prob)
    print("Test accuracy: {:.6f}".format(np.mean(test_acc)))
```
```python
# now reshape the probs array
probs = np.array(probs)
probs = probs.reshape((probs.shape[0]*probs.shape[1]), probs.shape[2])
y_test = y_test[:len(probs),:]
# model complete

# save the results
hf = h5py.File('h5files/' + resultsave, 'w')
hf.create_dataset('predict_valid', data=probs)
hf.create_dataset('y_test', data = y_test)
hf.close()

del(X_test, y_test, lab_ch_test)
```
````

---

<div className="flex flex-col space-y-1 mb-4">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Plot the predictions accuracy</h2>
</div>

```python
hf = h5py.File('h5files/' + resultsave, 'r')
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

---

<div className="flex flex-col space-y-1 mb-4">
      <h1 className="text-lg font-semibold">Training and testing the model</h1>
      <h2 className="text-lg font-semibold mt-4">Plot the predictions accuracy heatmap</h2>
</div>

<img src='/heatmap.png' className="mx-auto w-1/2" alt="Prediction Accuracy"/>

---