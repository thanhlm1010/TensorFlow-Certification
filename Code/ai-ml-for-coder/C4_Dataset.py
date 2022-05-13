import os
import tensorflow as tf
import tensorflow_datasets as tfds
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#mnist_data = tfds.load("fashion_mnist")
#for item in mnist_data:
#    print(item)
#mnist_train = tfds.load(name="fashion_mnist", split="train")
#assert isinstance(mnist_train, tf.data.Dataset)
#print(type(mnist_train))
#for item in mnist_train.take(1):
#    print(type(item))
#    print(item.keys())
#    print(item['image'])
#    print(item['label'])

#mnist_info, info = tfds.load(name="fashion_mnist", with_info="true")
#print(info)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()
#data = tf.keras.datasets.fashion_mnist # load data

(training_images, training_labels), \
(test_images, test_labels) = tfds.as_numpy(tfds.load('fashion_mnist',
                                       split=['train', 'test'],
                                       batch_size=-1,
                                       as_supervised=True))


training_images = training_images / 255.0 # nomalization (0,255)->(0,1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5,
          callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])