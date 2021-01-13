from getdataset import GetDataset
from face_detector import detect
from feature_vector_function import feature_vector
import os
import numpy as np
from cv2 import imread, imwrite

if not os.path.exists('./photos') or not os.path.exists('./audios'):
    dataset = GetDataset()
    dataset.setDatasetCSV('../avspeech_train.csv')
    dataset.setAudioOutputFolder("audios")
    dataset.setPhotoOutputFolder("photos")
    # At the real training, we must modify convertVideos
    dataset.convertVideos()

if not os.path.exists('./face_photos'):
    images = [f for f in os.listdir('./photos') if os.path.isfile(os.path.join('./photos', f))]
    os.mkdir(os.path.join("./", "face_photos"))
    for image in images:
        read_image = imread(os.path.join('./photos', image))
        face = detect(read_image)
        imwrite(os.path.join('./face_photos', image), face)
# Faces are saved with the same name of the original photos.
features = feature_vector(os.path.join('./face_photos', 'AvWWVOgaMlk.png')).reshape((4096, 1))
with open('weights.txt', mode='w') as f:
    for feature in features:
        f.write(str(feature.item()) + '\n')
    
# print(feature_vector(os.path.join('./face_photos', 'AvWWVOgaMlk.png')).reshape((4096, 1))[23])