import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json

batch_size = 32
image_size = 224

category_names = {}

def process_image(image):
    
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    
    return image.numpy()

def predict(img_path, model, top_k=5):
    top_k = int(top_k)
    
    if top_k < 1:
        top_k = 1
    
    categories = []
    probabilities = []
    
    img = Image.open(img_path)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    processed_image = process_image(img)
    probability_list = model.predict(processed_image)
    
    rank = probability_list[0].argsort()[::-1]
    
    for i in range(top_k):
        position = rank[i] + 1
        
        category_name = category_names[str(position)]
        probabilities.append(probability_list[0][position])
        categories.append(category_name)
        
    return probabilities, categories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names')
    args = parser.parse_args()
    
    img_path = args.arg1
    model_path = args.arg2
    
    top_k = args.top_k
    if top_k is None:
        top_k = 5
        
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    
    with open(args.category_names, 'r') as file_opener:
        category_names = json.load(file_opener)
    
    probabilities, categories = predict(img_path, model, top_k)
    
    for i in range(int(top_k)):
        print("Category: {},\nProbability: {}\n".format(categories[i], probabilities[i]))