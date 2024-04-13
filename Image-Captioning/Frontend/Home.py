import streamlit as st
from PIL import Image
import os
from pickle import load
from keras.models import load_model
import utils_transformer
import utils

xception_model = utils.cnn_model()
image_features_extract_model, transformer = utils_transformer.initialize_weights()

# Set up the Streamlit app
st.set_page_config(layout="wide")

st.title("Image Captioning")

st.markdown('#')

col1, col2 = st.columns(2)

with col1:
    option = st.selectbox(
        'Model of Choice?',
        ('Initial Model', 'Improvised Model'))

    st.subheader('You have selected: {}'.format(option))

with col2:
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    if option == 'Improvised Model':
        save_folder = './tmp/'+uploaded_file.name
        with open(save_folder, mode='wb') as w:
            w.write(uploaded_file.getvalue())

        description = utils_transformer.predict_caption(
            save_folder, image_features_extract_model, transformer)
        
        os.remove(save_folder)

    else:
        tokenizer = load(open("./models/tokenizer.p", "rb"))
        model = load_model('./models/model_7.h5')
        max_length = 32

        photo = utils.extract_features(image, xception_model)
        description = utils.predict_caption(
            model, tokenizer, photo, max_length)
        

    st.markdown('#')
    col1, col2, col3 = st.columns(3)

    with col2:
        st.subheader("Text extracted from the image: {}".format(description))
        st.image(image, caption='Uploaded Image', width=500)
    
