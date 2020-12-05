import tensorflow as tf

def Audio_Encoder():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Input(shape = (598,257,2)))

    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(1, 1), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(1, 1), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(128, (4, 4), strides=(1, 1), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2, 1)))

    model.add(tf.keras.layers.Conv2D(128, (4, 4), strides=(1, 1), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2, 1)))

    model.add(tf.keras.layers.Conv2D(128, (4, 4), strides=(1, 1), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2, 1)))

    model.add(tf.keras.layers.Conv2D(256, (4, 4), strides=(1, 1), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.MaxPool2D(pool_size=[2,1], strides=(2, 1)))

    model.add(tf.keras.layers.Conv2D(512, (4, 4), strides=(1, 1), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='VALID'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='VALID'))

    model.add(tf.keras.layers.AveragePooling2D(pool_size=(6,1),strides=1,padding="VALID"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Flatten())
	
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Dense(4096))	


    return model


model = Audio_Encoder()

print(model.summary())
