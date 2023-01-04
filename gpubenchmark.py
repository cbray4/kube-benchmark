import tensorflow as tf
import datetime
from tensorflow.keras.layers import *

start = datetime.datetime.now()
print("Started at: " + start.strftime("%Y-%m-%d %H:%M:%S"))

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float16') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64, drop_remainder=True).cache().prefetch(-1)

model = tf.keras.Sequential([        
    Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('softmax')
    ])

model.compile(loss='categorical_crossentropy', 
    optimizer=tf.optimizers.SGD(learning_rate=0.016), 
    metrics=['accuracy'], steps_per_execution=1)

model.fit(ds_train, epochs=1000, steps_per_epoch=781)

end = datetime.datetime.now()
print("Finished at: " + end.strftime("%Y-%m-%d %H:%M:%S"))
