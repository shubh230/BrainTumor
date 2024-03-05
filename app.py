import streamlit as st
import numpy as np

from io import BytesIO
from PIL import Image
import tensorflow as tf

st.header("Brain Tumor :red[Detection] :brain:",divider='rainbow')
with st.sidebar:
    upload_file = st.file_uploader("Choose a File")

if upload_file is not None:
    if st.button("Predict"):
        col1, col2 = st.columns(2)
        with col1:
            MODEL = tf.keras.models.load_model("Models/7")
            CLASS_NAMES = ["Normal","Glioma Tumor","Meningioma Tumor","Pitiutary Tumor"]
            st.image(upload_file,caption="Uploaded Image",width=300)
            img = np.array(Image.open(BytesIO(upload_file.getvalue())))
            img = tf.image.resize(img,[256,256])
            img_bt = np.expand_dims(img,0)
            prediction = MODEL.predict(img_bt)
            prediction_class = CLASS_NAMES[np.argmax(prediction[0])] 
            confidence = np.max(prediction[0])
        with col2:
            if(prediction_class == "Normal"):
                st.title(":green[No Tumor] :smiley:")
            else:
                tab1,tab2 = st.tabs(["🧠Result",f"📝{prediction_class}"])
                with tab1:
                    st.title(f":red[{prediction_class}] 😭")
                with tab2:
                    if prediction_class == "Glioma Tumor":
                        st.write("Glioma is a type of tumor that occurs in the brain and spinal cord. It is a type of tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function.")
                    elif prediction_class == "Meningioma Tumor":
                        st.write("Meningioma is a type of tumor that forms on the membranes that cover the brain and spinal cord just inside the skull. Specifically, the tumor forms on the three layers of membranes that are called meninges.")
                    else:
                        st.write("Pituitary tumors are abnormal growths that develop in your pituitary gland. Some pituitary tumors result in too many of the hormones that regulate important functions of your body. Some pituitary tumors can cause your pituitary gland to produce lower levels of hormones.")

