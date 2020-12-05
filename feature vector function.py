from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def feature_vector (picturename): #picturename like 'mug.jpg' with the single quates
    vgg16_model = VGG16()
    model = Sequential()

    for layer in vgg16_model.layers[:-1]:

        model.add(layer)

    for layer in model.layers:

        layer.trainable = False

    image = load_img(picturename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1,image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return model.predict(image)
