import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Nondies vs Cuea.txt')
Nondies = pd.read_csv('Nondies.csv')
st.set_page_config(layout="centered", initial_sidebar_state="expanded", page_title = "Player Performance Metrics")

st.sidebar.header("Menu")

menu=['Display Data', 'Graphs']
selections = st.sidebar.selectbox('',menu)

if selections == 'Display Data':
    st.subheader("Display Data")
    st.dataframe(data)
    
    if st.checkbox("Show data"):
        st.write("Data Shape: ")
        st.write('{} rows and {} columns'.format(data.shape[0],data.shape[1]))
        
        st.markdown("Descriptive statistics")
        st.write(data.describe())
    
if selections == 'Graphs':
    indexes =Nondies['Athlete']
    indexing = np.arange(len(indexes.index))
    s_indexes=[x.split(' ')[1] for x in indexes]
    width = 0.5
    
    col1, col2 = st.columns(2)
    metric1 = st.selectbox('Metric1', Nondies.columns[2:-1])
    with col1:
        plt.figure(figsize=[15,10])
        plt.bar(s_indexes, Nondies[metric1])
        plt.ylabel(metric1)
        plt.title(metric1 + ' Graph')
        plt.show()
        st.pyplot(plt)
            
    with col2:
        plt.figure(figsize=[15,10])
        sns.kdeplot(Nondies[metric1], shade = True)
        plt.title('Distribution of ' + metric1)
        st.pyplot(plt)
        
    if st.checkbox('Compare with another metric'):
        
    
        metric2 = st.selectbox('Metric2', Nondies.columns[2:-1])
    
    
        
    
        
        with col1:
            fig, ax1 = plt.subplots(figsize = (20,20))
            ax2 = ax1.twinx()

            ax1.bar(indexing + width,Nondies[metric1], width=width, color = '#e5ae38', label=metric1)
            ax2.bar(s_indexes, Nondies[metric2], width=width, color='#008fd5', label=metric2)
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
            sns.scatterplot( data= Nondies, x =metric1, y = metric2, hue = 'Athlete', markers = '+')
            plt.title(metric1 + ' vs ' + metric2)
            plt.legend(bbox_to_anchor=(1.1, 1))
            plt.grid(True)
            plt.show()
            st.pyplot(plt)
            
#     else:
#         with col1:
#             plt.bar(Nondies['Athlete'], Nondies[metric1])
#             plt.show()
            
#         with col2:
#             sns.kdeplot(Nondies[metric1], shade = True)
    
