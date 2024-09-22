import streamlit as st
from sdv.single_table import CTGANSynthesizer
import numpy as np
import torch
from model import AnomalyDetectorNet

features = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
       'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
       'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max',
       'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
       'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
       'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
       'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std',
       'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Len', 'Bwd Header Len',
       'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max',
       'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt',
       'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
       'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio',
       'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
       'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
       'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
       'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
       'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
       'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
       'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
       'Idle Min', 'Protocol_17', 'Protocol_6']

synthesizer = CTGANSynthesizer.load(
    filepath='weights/my_synthesizer.pkl'
)

model = AnomalyDetectorNet(no_of_features=72)
model.load_state_dict(torch.load('weights/detector.pth'))
synthetic_data = synthesizer.sample(num_rows=1)

st.title(" CTGAN IOT Anomaly detection")

ct_desc = """
    CTGAN is a collection of Deep Learning based synthetic data generators for single table data, 
         which are able to learn from real data and generate synthetic data with high fidelity.
"""

st.write(ct_desc)
st.image('images/ctgan_arc.png',caption="Flow")
st.image('images/ctgan.png',caption="CTGAN Report")

st.image('images/network.png',caption='Model')
st.image('images/network_2.png',caption="Network")
st.image('images/report.png',caption="Report of Detector Model")
st.image('images/loss.png',caption="Loss")
st.image('images/acc.png',caption="Accuracy")


gen_data = st.button("Generate Data")

if gen_data:
    st.write("## Generated Data")
    st.dataframe(synthetic_data)
    

predict_anomaly = st.button("Predict Anomaly")

if predict_anomaly:
    input_vars = synthetic_data[features]
    input_ar = np.array(input_vars)
    inputs_tensor = torch.from_numpy(input_ar.astype(np.float32))
    pred = model(inputs_tensor)
    pred = pred.round()
    anomaly = int(pred.item())
    
    if anomaly:
        st.write("# Anomaly Detected")
    else:
        st.write("# No Anomaly Detected")
else:
    st.write("Please Generate Data first")
    

