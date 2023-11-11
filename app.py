import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from tensorflow import keras
import plotly.express as px




def main():
    st.title('Mammogram 2nd Opinion')
    st.write('Upload Mammogram image that you need to classify')

    file = st.file_uploader('please upload an image')
    if file:
        image = Image.open(file)
        

        resized_image = image.resize((224,224))
        image_tf = tf.constant(resized_image)
        image_tf = tf.expand_dims( image_tf, axis =0)
        st.write(image_tf.shape)

        
        model = keras.models.load_model('E:\Mammogram_app\My_model.hdf5')

        pred_proba = model.predict(image_tf)


        prediction_classes = ['BIRAD_1' , 'BIRAD_3' , 'BIRAD_4' , 'BIRAD_5']
        prediction = prediction_classes[pred_proba.argmax()]
        with st.spinner('please wait..'):
                
                    
                #for i in range(101):
                    #st.progress(i)
                #do_something_slow()
                
                with st.container():
                    s1,s2 = st.columns([0.5,0.5])
                with s1:
                    st.image(image , use_column_width=False , )

                with s2:
                    
                    df = pd.DataFrame(data = pred_proba  , columns = prediction_classes)
                    st.dataframe(df)
                    #st.number_input( value = prediction , format = '%.1f')

                    st.text_input('Model prediction' , f'{prediction}')
                    
                    #st.markdown(f"Model Prediction  =  *{prediction}*")
                        

                    figure = px.bar(data_frame= df)
                    
                
                    st.plotly_chart(figure)

        


    else:
        st.text('You have not uploaded an image')

if __name__ == '__main__':
    main()

   

