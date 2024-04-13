import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications.xception import Xception
from keras.utils import pad_sequences
import argparse

def cnn_model():
  xception_model = Xception(include_top=False, pooling="avg")
  return xception_model

def extract_features(image, model):
  image = image.resize((299,299))
  image = np.array(image)
  # for images that has 4 channels, we convert them into 3 channels
  if image.shape[2] == 4: 
      image = image[..., :3]
  image = np.expand_dims(image, axis=0)
  image = image/127.5
  image = image - 1.0
  feature = model.predict(image)
  return feature

def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None


def predict_caption(model,tokenizer,image,maxlen=32):
  '''
  image.shape = (1,4462)
  '''
  index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])
  in_text = 'start'

  for iword in range(maxlen):
      sequence = tokenizer.texts_to_sequences([in_text])[0]
      sequence = pad_sequences([sequence],maxlen)
      yhat = model.predict([image,sequence],verbose=0)
      yhat = np.argmax(yhat)
      newword = index_word[yhat]
      in_text += " " + newword
      if newword == "end":
          break
  
  return_str = in_text.split(" ")
  return_str = return_str[1:]
  return_str = return_str[:-1]

  return " ".join(return_str)

