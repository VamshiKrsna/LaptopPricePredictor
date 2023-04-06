import streamlit as st
import numpy as np
import joblib

model = joblib.load('C:\\Users\\Rizen3\\Desktop\\vamshi\\Work\\Innomatics\\LaptopProject\\rfr_model.pkl')

st.title("Laptop Price Predictor")

os_map = {'Windows 11': 0, 'Windows 10': 1, 'MacOS': 2, '64 bit Chrome': 3, '64 bit DOS': 4}
storage_type_map = {'HDD': 0, 'SSD': 1}
processor_map = {'i5': 0, 'i3': 1, 'i7': 2, 'i9': 3, 'M1': 4, 'M2': 5, 'Ryzen 3': 6, 'Ryzen 5': 7, 'Ryzen 7': 8, 'Ryzen 9': 9, 'Legacy': 10}
ddr_map = {'DDR3':3,'DDR4':4,'DDR5':5}
storage_map = {256:256,512:512,1024:1024,2048:2,128:128}

ram_size = st.selectbox('RAM Size (in GB)', [4, 8, 16, 32])
ddr_version = st.selectbox('DDR Version', ['DDR3', 'DDR4','DDR5'])
storage = st.selectbox('Storage Capacity (in GB)', [256, 512, 1024, 128, 2048])
os_enc = st.selectbox('Operating System', ['Windows 11', 'Windows 10', 'MacOS', '64 bit Chrome', '64 bit DOS'])
storage_type = st.selectbox('Storage Type', ['HDD', 'SSD'])
processor = st.selectbox('Processor', ['i5', 'i3', 'i7', 'i9', 'M1', 'M2', 'Ryzen 3', 'Ryzen 5', 'Ryzen 7', 'Ryzen 9', 'Legacy'])

os_enc = os_map[os_enc]
storage_type = storage_type_map[storage_type]
processor = processor_map[processor]
ddr_version = ddr_map[ddr_version]
storage = storage_map[storage]

features = np.array([[ram_size, ddr_version, storage, os_enc, storage_type, processor]])

features = features.reshape(1, -1)

mrp = model.predict(features)
mrp = mrp.astype(int)

button_pred = st.button("Reveal Price")


if button_pred:
    st.write("## Predicted MRP: ₹{:.0f}".format(mrp[0]))