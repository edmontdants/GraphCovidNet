# GraphCovidNet
COVID-19 CT-scan and CXR image detection using GraphCovidNet, a GIN based architecture. The code has been performed on 2-class,3-class and 4-class datasets.

# Dataset link
1. SARS-COV-2 Ct-Scan Dataset, CT-scan, 2-class: https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset
2. COVID-CT Dataset, CT-scan, 2-class: https://github.com/UCSD-AI4H/COVID-CT
3. covid-chest-xray dataset: https://github.com/ieee8023/covid-chestxray-dataset + Chest X-Ray Images (Pneumonia): https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia , CXR, 3-class
4. CMSC-678-ML-Project, CXR, 3/4-class: https://github.com/vj2050/Transfer-Learning-COVID-19

# File Structure and Working procedure
1. First apply edge detection accroding to the class-number: Edge detection/edge_detection_<class-number>class.py
2. Then prepare graph-datasets using edge-preparation: Edge preparation/edge_preparation_<class-number>class.py
3. Finally edge preperation produces five kinds of dataset for graph classification:
  path name: .../GraphTrain/dataset/<dataset_name>/raw/<dataset_name>_<data_file>.txt
  <data_file> can be:
    1. A--> adjancency matrix 
    2. graph_indicator--> graph-ids of all node 
    3. graph_labels--> labels for all graph 
    4. node_attributes--> attribute(s) for all node 
    5. node_labels--> labels for all node
