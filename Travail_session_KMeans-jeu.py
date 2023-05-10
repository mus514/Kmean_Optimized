#!/usr/bin/env python
# coding: utf-8

# ### Librairies

# In[1]:


import pandas as pd
import numpy as np
from math import * 
from random import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  silhouette_score
import time
import datetime as datetime
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# ### Distance euclidienne

# In[2]:


def eucludian_dist(row,centroid):
    return sqrt(sum((row-centroid)**2))


# ### Calcul des valeurs des attributs des centroides

# In[3]:


def centroids_update(df_new,features_col,n_cluster): #dataframe=df_new
    
    df_new_centroid=pd.DataFrame()

    for clust in range(n_cluster):
        
        for col in features_col:
            df_new_centroid.loc[clust,col]=df_new.loc[df_new.cluster==clust,col].mean()
        df_new_centroid.loc[clust,'cluster']=clust
    
    df_new_centroid['cluster']=df_new_centroid['cluster'].map(int)
    df_new_centroid.dropna(axis=0,how='any',inplace=True)
    df_new_centroid.reset_index(inplace=True)
    df_new_centroid.drop(columns='index',inplace=True)
    return df_new_centroid
            
        
    


# ### Affectation de classes aux observations

# In[4]:


def assign_class(df,df_centroid,features_col):
    df_new_columns=pd.DataFrame()
    #df_centroid['cluster']=None
    for i in range(len(df)):
        distances=[]
        clusters=[]
        for j in range(len(df_centroid)):
        
            row=np.array(df.loc[i,features_col])
            centroid=np.array(df_centroid.loc[j,features_col])
            dist=eucludian_dist(row,centroid)
            clust=df_centroid.loc[j,'cluster']
            distances.append(dist)
            clusters.append(clust)
        df_new_columns.loc[i,'distance_min']=min(distances)
        df_new_columns.loc[i,'cluster']=clusters[distances.index(min(distances))]

    df_new_columns['cluster']=df_new_columns['cluster'].map(int)
    df_new_columns['distance_min']=df_new_columns['distance_min'].map('{:.3f}'.format)
    df_new=pd.concat([df,df_new_columns],axis=1)
    return df_new


# ### Verification de la stabilite des centroides

# In[5]:


def centroid_change(df_centroid,df_new_centroid,features_col,epsilon):
    
    if (len(df_centroid)==len(df_new_centroid)) and list(df_centroid.columns)==list(df_new_centroid.columns):
        df_gap=(df_new_centroid[features_col]-df_centroid[features_col])**2
        difference=[]
        for i in range(len(df_gap)):
            gap_value=sqrt(np.array(df_gap.loc[i,:]).sum())
            difference.append(gap_value)
        
            
        if (np.array(difference)>epsilon).sum()>0:
            centroids_changes=True
        else:
            centroids_changes=False
        
        return centroids_changes
    else: 
        print ('Error: Check dimensions and columns of dataframes')
        return None
    


# ### Processus Kmeans

# In[6]:


def kmeans(df,df_centroid,features_col,n_cluster,max_iter,epsilon):
    iter=0
    change=True
    while((iter<=max_iter) & (change!=False)):
        df_new=assign_class(df,df_centroid,features_col)
      
        df_new_centroid=centroids_update(df_new,features_col,n_cluster)
      
        change=centroid_change(df_centroid,df_new_centroid,features_col,epsilon)
        df_centroid=df_new_centroid
        iter+=1
    return df_new,df_new_centroid


# ### Processus VNS adapté à Kmeans

# In[7]:


def pertmerge(dataframe,df_centroid,n_cluster,features_col,percent=0.5): #dataframe=df_new
    
    cluster1=randint(0,n_cluster-1)
   
    df1=dataframe.loc[dataframe.cluster==cluster1,features_col].copy()
    
    idx1=np.random.randint(0,len(df1)-1)
     
    df_other_centroids=df_centroid[df_centroid.cluster!=cluster1].copy()
    
    cluster=[]
    distance=[]
    row=np.array(df1.iloc[idx1,:])
    
    for i in range(len(df_other_centroids)):
        centroid=np.array(df_other_centroids.iloc[i,:-1])
        clust=df_other_centroids.iloc[i,-1]
        
        dist=eucludian_dist(row,centroid)
        cluster.append(clust)
        distance.append(dist)
    
    dict={'cluster':cluster,'dist':distance}
    
    cluster_distance=list(dict.values())
    
    closest_cluster=cluster_distance[0][cluster_distance[1].index(min(cluster_distance[1]))]
    
    df2=dataframe[dataframe.cluster==closest_cluster].copy()
    
    # Selection des elements a transfert entre les deux clusters 1 et 2
    n1=round(percent*len(df1))
    n2=round(percent*len(df2))
    
    list_elements_1=list(sample(list(range(len(df1))),n1))
    list_elements_2=list(sample(list(range(len(df2))),n2))
    
    df1_cluster=dataframe.loc[dataframe.cluster==cluster1,features_col+['cluster']].copy()
    df2_cluster=dataframe.loc[dataframe.cluster==closest_cluster,features_col+['cluster']].copy()
    
    
    df1_cluster.iloc[list_elements_1,-1]=int(closest_cluster)
    df2_cluster.iloc[list_elements_2,-1]=int(cluster1)

    
    df_otherClusters=dataframe[(dataframe.cluster!=cluster1) & (dataframe.cluster!=closest_cluster)].copy()
    df_2clusters=pd.concat([df1_cluster,df2_cluster],axis=0,ignore_index=False).sort_index()
    
    # Calcul des centroids des 2 clusters modifiés
    df_centroid_otherClusters=df_centroid[(df_centroid.cluster!=cluster1) & (df_centroid.cluster!=closest_cluster)].copy()
    df_centroid_2clusters=df_centroid[(df_centroid.cluster==cluster1) | (df_centroid.cluster==closest_cluster)].copy()  
    df_centroid_2clusters=centroids_update(df_2clusters,df_centroid_2clusters,n_cluster)
   
    return df_otherClusters,df_2clusters,df_centroid_otherClusters,df_centroid_2clusters
        
        
    


# In[8]:


#Condition d'arret
# Iteration max
# Stabilité des centroides


# ## Implementation Kmeans & VNS
# 
# 
# 

# In[9]:


# Lecture et chargement des donnees
data=pd.read_csv("jeu.txt",sep=',',header=0,index_col=None)
data.head()


# In[10]:


# Dimensions du jeu de donnees
data.shape


# In[11]:


#Selection aleatoire d'un echantillon de 1000 observations
sample_idx=list(sample(list(range(len(data))),1000))
df=data.iloc[sample_idx,:].reset_index()
df.drop('index',axis=1,inplace=True)


# In[12]:


# Verification de la correlation des deux variables 
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(df.corr(), annot=True, linewidths=.2, fmt= '.1f',ax=ax);


# In[13]:


import os
os.system(f"pip install -U seaborn")
import seaborn as sns


# In[126]:


# Histogrammes des distributions des observations par rapport aux variables
dims=(1,2)
figure,axes=plt.subplots(dims[0],dims[1],figsize=(15,5))
axis_i=0
for i in list(df.columns):
    
    sns.histplot(df,x=i,ax=axes[axis_i])
    plt.xticks(rotation=0)
    axis_i+=1
    plt.suptitle('Histogrammes des observations')
img_file="features_histograms.png"
plt.savefig(img_file)


# In[ ]:





# In[16]:


datetime.datetime.now()


# #### Execution de k-means pour tester une selecion de  nombres de clusters

# In[17]:



epsilon=1e-6
max_iter=300
features_col=list(df.columns)
max_cluster=20
S_score_max=-1
df_n_centroid=pd.DataFrame()
best_n_cluster=None
cluster_score=[]
df_clusters_centroids=pd.DataFrame()

for n_cluster in range(2,max_cluster+1):
    

    clusters_array=np.array([i for i in range(n_cluster)])
    centroid_index=list(sample(list(range(len(df))),n_cluster))
    
    df_sample=df.iloc[centroid_index,:]
    df_sample.reset_index(inplace=True)
    df_centroid=df_sample.iloc[:,1:].copy()
    df_centroid['cluster']=clusters_array
   
    
    start_time=time.time()
    df_new,df_new_centroid=kmeans(df,df_centroid,features_col,n_cluster,max_iter,epsilon)
    end_time= time.time()
    exec_time=end_time-start_time
   
    S_score = silhouette_score(X=df_new[features_col], labels=df_new.iloc[:,-1])
    print(n_cluster,' : ',S_score)
    
    
    
    cluster_score.append({'n_cluster':n_cluster,'silhouette':S_score,'exec_time':exec_time})
    df_new_centroid['num_cluster']=n_cluster
    df_clusters_centroids=pd.concat([df_clusters_centroids,df_new_centroid],axis=0,ignore_index=True)
    


df_cluster_score=pd.DataFrame.from_dict(cluster_score)
df_cluster_score.sort_values(by='silhouette',axis=0,ascending=False,ignore_index=True,inplace=True)
print(df_cluster_score)
         
    
    


# In[18]:


datetime.datetime.now()


# In[19]:


# Verification de l'attribution de classes aux observations
df_new.tail()


# In[ ]:





# In[21]:


datetime.datetime.now()


# #### Execution de 100 iterations de VNS pour chacun des topN de nombre de clusters

# In[22]:


#Top N clusters numbers

topN=5
kmax=100

df_topN_clusters_centroids=pd.DataFrame()
df_topN_clusters=pd.DataFrame()
k_cluster_score=[]
for n_cluster in list(df_cluster_score.loc[:(topN-1),'n_cluster']):
    
    S_score_ref=df_cluster_score.loc[df_cluster_score.n_cluster==n_cluster,'silhouette'].values[0]
    S_score_max=S_score_ref
    duration=df_cluster_score.loc[df_cluster_score.n_cluster==n_cluster,'exec_time'].values[0]
    k_cluster_score.append({'num_cluster':n_cluster,'k':0,'silhouette':S_score_ref,'duration':duration})
    
    df_new_centroid=df_clusters_centroids[df_clusters_centroids.num_cluster==n_cluster].copy()
    df_new_centroid.reset_index(inplace=True)
    df_new_centroid.drop(columns=['index','num_cluster'],inplace=True)
    df_new=assign_class(df,df_new_centroid,features_col)
    df_best_cluster=df_new.copy()
    df_best_centroid=df_new_centroid.copy()
    

    print("REFERENCE using only initial classic Kmeans: For n_clusters =",
          n_cluster,"The average silhouette_score is :",S_score_ref)

    k=1
    while k<=kmax:
      
        percent=sample(list(np.linspace(0.5,0.9,9)),1)[0]
        
        start=time.time()
        df_otherClusters,df_2clusters,df_centroid_otherClusters,df_centroid_2clusters=pertmerge(dataframe=df_new,
                                                                                            df_centroid=df_new_centroid,
                                                                                            n_cluster=n_cluster,
                                                                                            features_col=features_col,
                                                                                            percent=percent)
    
    
        df_2clusters.drop(columns=['cluster'],inplace=True)
        df_2clusters.reset_index(inplace=True)
    
        df_centroid_2clusters.reset_index(inplace=True)
        df_centroid_2clusters.drop(columns=['index'],inplace=True)
    
    

        df_new_2clusters,df_new_centroid_2clusters=kmeans(df_2clusters,df_centroid_2clusters,
                                                       features_col,n_cluster,max_iter,epsilon)
        end=time.time()
        duration=end-start
    
        df_new_2clusters.set_index('index',inplace=True)
    
    
        df_joined=pd.concat([df_otherClusters,df_new_2clusters],axis=0,ignore_index=False).sort_index()
   
        df_centroid_joined=pd.concat([df_centroid_otherClusters,df_new_centroid_2clusters],axis=0,ignore_index=True)
    
        S_score = silhouette_score(X=df_joined[features_col], labels=df_joined.iloc[:,-1])

    
    
        df_new=df_joined.copy()
        df_new_centroid=df_centroid_joined.copy()
    
        if S_score>S_score_max:
            S_score_max=S_score
            df_best_centroid=df_new_centroid.copy()
            df_best_cluster=df_new.copy()
        
        k_cluster_score.append({'num_cluster':n_cluster,'k':k,'silhouette':S_score,'duration':duration})
        k+=1
    
    df_best_centroid['num_cluster']=n_cluster
    df_best_cluster['num_cluster']=n_cluster
    
    
    df_topN_clusters_centroids=pd.concat([df_topN_clusters_centroids,df_best_centroid],axis=0,ignore_index=True)
    df_topN_clusters=pd.concat([df_topN_clusters,df_best_cluster],axis=0,ignore_index=True)
    
    print("The best silhouette_score is :",S_score_max,' with a number of clusters of: ',n_cluster)
    print("\n") 

df_k_cluster_score=pd.DataFrame.from_dict(k_cluster_score)


# In[23]:


datetime.datetime.now()


# In[24]:


# Centroides des clusters pour differents nombres de clusters
df_topN_clusters_centroids


# In[25]:


# Clusters assignés à chaque observaion pour différents nombres de clusters
df_topN_clusters


# In[26]:


# Evolution des scores et des temps d'execution à chaque k itération VNS
df_k_cluster_score


# #### Visualisation des clusters issus de VNS

# In[127]:


dims=(2,3)
figure,axes=plt.subplots(dims[0],dims[1],figsize=(20,15))
axis_i,axis_j=0,0
sns.set(font_scale=1.25)
for n_cluster in list(df_cluster_score.loc[:(topN-1),'n_cluster']):

    sns.scatterplot(data=df_topN_clusters[df_topN_clusters.num_cluster==n_cluster],x="col1", y="col2", hue="cluster", size=None,
                    style=None, palette='tab10', hue_order=None, hue_norm=None, sizes=None,
               size_order=None, size_norm=None, markers=True, style_order=None, legend='auto', ax=axes[axis_i,axis_j])
    axes[axis_i,axis_j].set_title(str(n_cluster)+" clusters")
    plt.suptitle('Clusters - VNS')
    plt.xticks(rotation=0)
  
    axis_j+=1
    if axis_j==dims[1]:
        axis_i+=1
        axis_j=0

img_file="clusters_vns.png"
plt.savefig(img_file)


# #### Visualisation des clusters pour chacune des variables

# In[128]:


# Clusters

dims=(2,3)
figure,axes=plt.subplots(dims[0],dims[1],figsize=(20,15))
axis_i,axis_j=0,0

sns.set(font_scale=1.5)
for n_cluster in list(df_cluster_score.loc[:(topN-1),'n_cluster']):
    
    data_clusters = pd.melt(df_topN_clusters.loc[df_topN_clusters.num_cluster==n_cluster,features_col+['cluster']],id_vars="cluster",
                    var_name="features",
                    value_name='value')
    sns.stripplot(x="features", y="value", hue="cluster", data=data_clusters,ax=axes[axis_i,axis_j],palette='tab10')
    axes[axis_i,axis_j].set_title(str(n_cluster)+" clusters")
    plt.xticks(rotation=0)
    plt.title("N_clusters: "+str(n_cluster))
    plt.suptitle('Répartition des clusters-VNS pour chacune des variables')
    axis_j+=1
    if axis_j==dims[1]:
        axis_i+=1
        axis_j=0
img_file="clusters_vns_variables.png"
plt.savefig(img_file)


# #### Visualisation des positions des centroides pour chaque variable

# In[129]:


# Centroids
dims=(2,3)
figure,axes=plt.subplots(dims[0],dims[1],figsize=(20,15))
axis_i,axis_j=0,0
sns.set(font_scale=1.2)
#axis_i=0
for n_cluster in list(df_cluster_score.loc[:(topN-1),'n_cluster']):
    data_centroids = pd.melt(df_topN_clusters_centroids.loc[df_topN_clusters_centroids.num_cluster==n_cluster,features_col+['cluster']]
                         ,id_vars="cluster",
                    var_name="features",
                    value_name='value')
    sns.lineplot(x="features", y="value", hue="cluster", data=data_centroids,ax=axes[axis_i,axis_j],palette='tab10')
    
    axes[axis_i,axis_j].set_title(str(n_cluster)+" clusters")
    plt.suptitle('Position des centroides - VNS')
    plt.xticks(rotation=0)
    #axis_i+=1
    axis_j+=1
    if axis_j==dims[1]:
        axis_i+=1
        axis_j=0
img_file="centroids_positions_vns.png"
plt.savefig(img_file)


# #### Evolution du silhouette_score à chaque itération VNS

# In[130]:


# Scores evolution as per k_vns & n_cluster

data_k_vns = df_k_cluster_score.pivot("k","num_cluster","silhouette")

plt.figure(figsize=(15,5))
sns.lineplot(data=data_k_vns,palette="tab10")
plt.title('Silhouette score pour les itérations VNS')
plt.xticks(rotation=0)
img_file="silhouette_vns.png"
plt.savefig(img_file)


# #### Evolution du temps d'execution à chaque itération VNS

# In[131]:


# Execution time
plt.figure(figsize=(15,5))
sns.lineplot(data=df_k_cluster_score,x="k",y="duration",hue="num_cluster",palette='tab10')
plt.title('Temps d\'éxecution à chaque itération VNS')
plt.xticks(rotation=0)
img_file="exec_time_vns.png"
plt.savefig(img_file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




