import numpy as np
import pandas as pd
import pandas as sqrt
from numpy.lib.function_base import average
from numpy import sqrt
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
import tqdm
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import time
import streamlit as st

#---------------------------------------------------------------------------------------------------------------------#
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#thêm thư viện linear_kernel
#core here
from sklearn.metrics.pairwise import linear_kernel

new_user = pd.read_csv('data_users.csv')
new_job = pd.read_csv('data_jobs.csv')

new_job.fillna('không',inplace=True)
new_user.fillna('không',inplace=True)

new_job.rename(columns={"Unnamed: 0": "JobID"},inplace= True)
new_user.drop(columns='Unnamed: 0',inplace=True)


job = new_job['Industry'].to_list()
user = new_user['Industry'].to_list()

corpus_merge  = job + user

def get_recommendation(userid, location, num_jobs):
      

  # Create new job dataframe by location
  df_nex_job =  new_job[new_job['Job Address']==location].reset_index(drop=True)
  df_nex_job['JobID']=[x for x in range(len(df_nex_job))]
  #job_matrix =  encoding(df_nex_job['Job Requirements'])
  #job_matrix= vectorizer.fit_transform(df_nex_job['Industry'])
  vectorizers = CountVectorizer()
  Y = vectorizers.fit_transform(corpus_merge)
  Vocabulary = vectorizers.get_feature_names_out()
  vectorizer = TfidfVectorizer(max_features= 617,vocabulary= Vocabulary)
  overview_matrix = vectorizer.fit_transform(df_nex_job['Industry'])
  overview_matrix1 = vectorizer.fit_transform(new_user['Industry'])
  cosine_sim = linear_kernel(overview_matrix1, overview_matrix)
  #Caculate matrix.
  #cosine_sim = linear_kernel(overview_matrix1, job_matrix)


  # get index by user id 
  v = new_user[new_user['UserID'] == userid ].index.values.astype(int)[0]
  # Sắp xép dựa trên điểm số tương tự
  sim_scores = list(enumerate(cosine_sim[v]))

  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

  sim_scores = sim_scores[0:num_jobs]

  job_indices = [i[0] for i in sim_scores]

  df = df_nex_job[df_nex_job['JobID'].isin(job_indices) ].reindex(job_indices)

  return df

#---------------------------------------------------------------------------------------------------------------------#

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

with st.sidebar:
    add_userID = st.number_input('Enter User Id:')
    with st.form('form1'):
        add_password = st.text_input('Enter password:')
        st.form_submit_button('Enter')
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

st.title("Jobs Recommendation System")
st.header("Welcome to Demo")
#ten = st.number_input("Enter your userID: ")
time.sleep(2)
#st.write('UserID: ',ten)

location = st.text_input("Enter the place: ")

st.write('Location: ',location)

#click = st.button('Search')
#location = int(location)
time.sleep(5)
df_recommendation = get_recommendation(add_userID,location,5)

for i in range(len(df_recommendation)):
    col1,col2 = st.columns(2)
    with col1:
        st.image('hinh'+str(i)+'.png',caption = '')
        #st.markdown(f'**Name Hotel**: {list_recommendations_content[i][0]}')
        
    with col2:
        st.markdown(f'**Job_title**: {df_recommendation.iloc[i,2]}')
        st.markdown(f'**Industry**: {df_recommendation.iloc[i,10]}')
        st.markdown(f'**Salary**: {df_recommendation.iloc[i,9]}')
        st.markdown(f'**Address**: {df_recommendation.iloc[i,6]}')
        st.markdown(f'**Job Description**: {df_recommendation.iloc[i,3][:100]}...')
        st.markdown(f'[Go to Website]({df_recommendation.iloc[i,1]})')



