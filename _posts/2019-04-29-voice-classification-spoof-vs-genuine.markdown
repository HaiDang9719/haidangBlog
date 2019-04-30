---
layout: post
title: "Voice classification spoof vs genuine"
image: introduction-to-ml.png
author: Hai Dang
comments: true
---
# Welcome to the voice classification spoof vs genuine
### Pipeline: 

INPUT -> FRONT-END -> BACK-END -> OUTPUT 

### Input:
* The number of wav audio file have been divided into three data sets train, develop and evaluation set.

### Front-end: Audio pre-processing:
* FFT vs CQT/CQCCs:
  * One of another traditional way is used (Fast) Fourier Transform (FFT). This technique is extremely powerful in time-frequency analysis, however it may lack frequency resolution at lower frequencies and temporal resolution at higher frequencies. In the other hand, one of efficient methods had been found in ASV 2015 for that problem is constant Q transform (CQT). The difference is that FFT imposes the regular spaced frequency bins while CQT employs geometrically spaced frequency bins, so CQT can across the entire spectrum and then get the a higher frequency resolution at lower frequencies and higher temporal resolution at higher frequencies. With this technique, it reflects more precisely the human perception system. Additionally, the baseline of feature extraction is proposed from ASV 2015 has shown that it will be more efficient to combine the CQT with traditional cepstral analysis called constant Q cepstral coefficients (CQCCs).
   * Q factor is a measure of selectivity of each filter and is defined as a ratio between center frequency \\(f_{k}\\) and bandwidth \\(\delta f\\):
    \\[Q=\frac{f_{k}}{\delta f}\\]
* Mel-spectrogram: 
  {% highlight python %}  
  def windows(data, window_size):
      start = 0
      while start < len(data):
          yield int(start), int(start + window_size)
          start += (window_size / 2)
  {% endhighlight %}
   * Each audio file has different lengths, the "windows" will slice the audio file into multiple small signals which have the same length to make it more convenient and more precisely to process. 
  {% highlight python %}  
  def extract_features(parent_dir,file_ext="*.wav",bands = 256, frames = 51):
      window_size = 512 * (frames - 1)
      log_specgrams = []
      labels = []
      i = 0
      for fn in glob.glob(os.path.join(parent_dir, file_ext)):
          sound_clip,s = librosa.load(fn, sr = 16000)
          label = train[train['fname'] == fn.split('/')[1]].identifier.values
          for (start,end) in windows(sound_clip,window_size):
              if(len(sound_clip[start:end]) == window_size):
                  signal = sound_clip[start:end]
                  melspec = librosa.feature.melspectrogram(signal, sr = 16000, n_mels = bands)
                  logspec = librosa.amplitude_to_db(melspec)
                  logspec = logspec.T.flatten()[:, np.newaxis].T
                  log_specgrams.append(logspec)
                  labels.append(label)
      log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
      features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
      for i in range(len(features)):
          features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
      return np.array(features), np.array(labels,dtype = np.str) 
  {% endhighlight %}
  * The method is used to convert the signal to frequency is Mel-spectrogram. Mel-spectrogram is visual representation of frequencies over time. To convert a sound to an spectrogram, we apply a filter bank representation of this sound. A sound file is divided into multiple short-time frames according to a certain window_length and time step. The signal changes overtime, thus to get the frequencies from signal, we need to applied Fourier Transform on each short time frames with the assumption that signal does not change at that period of time, otherwise it makes no sense, and the result is an approximation of the frequencies contours by concatenating the whole frames. Each frames will be apply window function to counteract the assumption made by the FFT that data is infinite and to reduce spectral leakage. Then, we can now do an N-point FFT into each frames to calculate the frequency spectrum:
  \\[P=\frac{\left|F F T\left(x_{i}\right)\right|^{2}}{N} \\]
  * Then, the frequency can be converted to Mel scale and vice versa through these equations: 
  \\[m=2595 \log _{10}\left(1+\frac{f}{700}\right)\\]
  \\[f=700\left(10^{m / 2595}-1\right)\\]
  * Those steps above have been made by default of the framework **librosa.feature.melspectrogram**. There are multiple parameters (y=None, sr=22050, S=None, n_fft=2048, hop_length=512, power=2.0, **kwargs) have been used in this framework. These files in the data set have the sampling rate is 16kHz, so *sr* need to be changed into 16000 instead of default value of 22050. Then the spectrogram is convert from amplitude to db and flattened. As a result, the log spectrogram is created in shape of (?, 60, 41, 1). Then we change the shape of log spectrogram into the shape of (?, 60, 41, 2) which is the input matrix vector for the CNN model.
  {% highlight python %}  
parent_dir = 'ASVspoof2017_V2_eval'
features,labels = extract_features(parent_dir)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(array(labels))
onehot_encoder = OneHotEncoder(sparse=False,categories='auto',)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
labels = onehot_encoder.fit_transform(integer_encoded)
      {% endhighlight %}
  * The goal of this project is to identify spoof vs genuine voice. So, we define two labels for the data set and encode it with OneHot Encoder from python to get the same shape with feature extraction above (?, 2).
### Back-end: Deep learning model CNN
  {% highlight python %}  
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)
    {% endhighlight %}
* The method named weight_variable and bias_variable will return Tensorflow variable of defined shapes, where bias variable is initialized with all ones and weight variable with zero mean and standard deviation of 0.1.
  {% highlight python %}  
def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')

def apply_convolution(x,kernel_size,num_channels,depth):
    weights = weight_variable([kernel_size, kernel_size, num_channels, depth])
    biases = bias_variable([depth])
    return tf.nn.relu(tf.add(conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1], 
                          strides=[1, stride_size, stride_size, 1], padding='SAME')
    {% endhighlight %}
* The Conv2d method is just a wrapper over Tensorflow conv2d function. It will be called by apply_convolution function, which takes input data, kernel/filer size, a number of channels in the input and output depth or number of channels in the output. It then gets weight and bias variables, applies convolution, adds the bias to the results and finally applies non-linearity (RELU). Max-Pooling can be applied using apply_max_pool function. It takes input data (usually output of convolution layer), kernel and stride size. 
  {% highlight python %}  
frames = 41
bands = 60

feature_size = 2460 #60x41
num_labels = 2
num_channels = 2

batch_size = 50
kernel_size = 30
depth = 16
num_hidden = 200

learning_rate = 0.01
training_iterations = 2000
    {% endhighlight %}
* Here, we define some configuration parameters for the deep learning model with Convolutional Neural Network as kernel size, total iterations, a number of neurons in the hidden layer, etc.
  {% highlight python %}  
X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

cov = apply_convolution(X,kernel_size,num_channels,depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
    {% endhighlight %}
* Tensorflow placeholder for input and output data are defined next. A convolution function is applied with a filter size of 30 and depth of 16 (number of channels, we will get as output from convolution layer). Next, the convolution output is flattened out for the fully connected layer input. There are 200 neurons in the fully connected layer as defined by the above configuration. The Sigmoid function is used as non-linearity in this layer. Lastly, the Softmax layer is defined to output probabilities of the class labels.    
{% highlight python %}  
loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    {% endhighlight %}
* The negative log-likelihood cost function will be minimised using Adam optimizer, the code provided below initialize cost function and optimizer. Also, define the code for accuracy calculation of the prediction by model.
    {% highlight python %}  
  cost_history = np.empty(shape=[1],dtype=float)
  with tf.Session() as session:
      tf.initialize_all_variables().run()
  
      for itr in range(total_iterations):    
          offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
          batch_x = tr_features[offset:(offset + batch_size), :, :, :]
          batch_y = tr_labels[offset:(offset + batch_size), :]
          
          _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
          cost_history = np.append(cost_history,c)
      
      print('Test accuracy: ',round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}) , 3))
      fig = plt.figure(figsize=(15,10))
      plt.plot(cost_history)
      plt.axis([0,total_iterations,0,np.max(cost_history)])
      plt.show()
      {% endhighlight %}
* Now the following code will train the CNN model using a batch size of 50 for 2000 iterations. After the training, it classifies testing set and prints out the achieved accuracy of the model along with plotting cost as a function of a number of iterations.
* The accuracy achieved around 90%.

Source: 

[Speech processing](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)

[Band filtering in the frequency domain](http://www.fon.hum.uva.nl/praat/manual/band_filtering_in_the_frequency_domain.html)

[urban sound classification - part 1](http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/)

[urban sound classification - part 2](http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/)

[Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)

[Mel-Spectrogram](https://www.mathworks.com/help/audio/ref/melspectrogram.html)

[Mel Frequency Cepstral Coefficient (MFCC)](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)

