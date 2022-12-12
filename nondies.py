import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from st_aggrid import AgGrid

path1 = 'Clusters Nondies vs CUEA.txt'
path2 = 'Clusters Nondies vs Impala.txt'
path3 = 'Clusters Nondies vs KCB.txt'
path4 = 'Clusters Nondies vs USIU.txt'
path5 = 'Clusters Nondies vs Kisumu.txt'

paths = [path1,path2,path3,path4]

# Nondies = pd.read_csv('Nondies.csv')
st.set_page_config(layout="centered", initial_sidebar_state="expanded", page_title = "Player Performance Metrics")

st.sidebar.header("Menu")
data = st.sidebar.selectbox(" Select Match", paths)



if data is not None:
    df = pd.read_csv(data)
    
else:
    df = pd.read_csv(path1)

menu=['Display Data', 'Graphs']
selections = st.sidebar.selectbox('',menu)

if selections == 'Display Data':
    st.subheader("Display Data")
    AgGrid(df)
    
    if st.checkbox("Show data"):
        st.write("Data Shape: ")
        st.write('{} rows and {} columns'.format(df.shape[0],df.shape[1]))
        
        st.markdown("Descriptive statistics")
        st.write(df.describe())
    
if selections == 'Graphs':
    
    def convert_time(time):
        time = time.split(':')
        x=time[0]
        y=time[1]
    
        return int((int(x) * 60) + int(y))
    
    df['Duration Total (min:sec)']=df['Duration Total (min:sec)'].apply(convert_time)
    df['Duration Speed Hi-Inten (min:sec)']=df['Duration Speed Hi-Inten (min:sec)'].apply(convert_time)
    
    
    indexes =df['Athlete']
    indexing = np.arange(len(indexes.index))
    s_indexes=[x.split(' ')[1] for x in indexes]
    width = 0.5
    
    col1, col2 = st.columns(2)
    metric1 = st.selectbox('Metric1', df.columns[2:-2])
    with col1:
        plt.figure(figsize=[15,10])
        plt.bar(s_indexes, df[metric1])
        plt.ylabel(metric1)
        plt.title(metric1 + ' Graph')
        plt.show()
        st.pyplot(plt)
            
    with col2:
        plt.figure(figsize=[15,10])
        sns.kdeplot(df[metric1], shade = True)
        plt.title('Distribution of ' + metric1)
        st.pyplot(plt)
        
    if st.checkbox('Compare with another metric'):
        
    
        metric2 = st.selectbox('Metric2', df.columns[2:-2])
    
    
        
    
        
        with col1:
            fig, ax1 = plt.subplots(figsize = (20,20))
            ax2 = ax1.twinx()

            ax1.bar(indexing + width,df[metric1], width=width, color = '#e5ae38', label=metric1)
            ax2.bar(s_indexes, df[metric2], width=width, color='#008fd5', label=metric2)
            ax1.set_title(metric1 + ' vs ' + metric2)
            ax1.set_ylabel(metric1)
            ax2.set_ylabel(metric2)
        
            ax1.grid(True)
            ax2.grid(True)
            fig.legend()
            plt.show()
            st.pyplot(plt)
        
        with col2:
            plt.figure(figsize=[20,20])
            sns.scatterplot( data= df, x =metric1, y = metric2, hue = 'Athlete', markers = '+')
            plt.title(metric1 + ' vs ' + metric2)
            plt.legend(bbox_to_anchor=(1.1, 1))
            plt.grid(True)
            plt.show()
            st.pyplot(plt)
            
# if selections == 'Machine Learning':
# #     from sklearn.cluster import KMeans
    
#     st.write('Using Machine Learning, we can group the athletes based on the available data.')
#     st.write('Players with statistics closer to each other are clumped together.')
#     st.write('We can first try to figure out how many clusters would be ideal, using the Elbow Method.')

#     def convert_time(time):
#         time = time.split(':')
#         x=time[0]
#         y=time[1]
    
#         return int((int(x) * 60) + int(y))
    
#     df['Duration Total (min:sec)']=df['Duration Total (min:sec)'].apply(convert_time)
#     df['Duration Speed Hi-Inten (min:sec)']=df['Duration Speed Hi-Inten (min:sec)'].apply(convert_time)
    
    
#     def position_codes(position):
#         if position == 'Forward':
#             return 0
#         else:
#             return 1
    
#     df[' Position'] = df[' Position'].apply(position_codes)
#     import sklearn
#     from sklearn.clusters import KMeans
    
# #     import joblib
# #     from joblib import load
# #     model = load(filename='perfomance_metric.joblib')
# #     model.predict(df[2:])
#     wcss = []
#     for k in range(1, 11):
#         km= KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
#         km.fit(df[2:])
#         wcss.append(km.inertia_/1000000)
    
#     plt.plot(range(1, 11), wcss)
#     plt.title('The Elbow Method', fontsize = 20)
#     plt.xlabel('No. of Clusters')
#     plt.ylabel('wcss')
#     plt.show()
#     st.pyplot(plt)

    
