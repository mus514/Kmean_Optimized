"""
Auteurs: Mustapha Bouhsen
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

# Importation du jeu de donnees
X, y = np.load('X.npy'), np.load('y.npy')
jeu = {'col1': X, 'col2': y}
jeu = pd.DataFrame(jeu)


def dist(x, y):
    """
    Demande à l'utilisateur d'enter les deux points pour calculer leur distance.

    Args:
            x : l'observation x
            y : l'observation y

    Returns:
        une valeur numerique: distance euclidienne
    """
    return np.sqrt(sum((x - y) ** 2))


def classe_(data, means, classes):
    """ fonction qui assigne aux observations la classe en se basant sur la distance minimale.

        Args:
            data : le jeu de donnees
            means : tabeau des centroides
            classes : tableau de 0 a k-1

        Returns:
            kl: classe pour chaque observation
    """
    kl = []
    means = np.array(means)
    data_ = data.copy().reset_index(drop=True)
    data_.reset_index()
    for i in range(data_.shape[0]):
        x = []
        for j in means:
            x.append(dist(data_.loc[i], j))
        kl.append(classes[x.index(min(x))])
    return kl


def paral_data(data, proc, mean, classe):
    """ fonction qui parallelise la fonction classe_

        Args:
            data : le jeu de donnees
            proc : nombre de processeur a utiliser
            mean : les centroide
            classe : tableau de 0 a k-1

        Returns:
            result: classe pour chaque observation
    """
    d_split = np.array_split(data, proc)
    with mp.Pool(proc) as pool:
        fn_ = partial(classe_, means=mean, classes=classe)
        result = np.concatenate(pool.map(fn_, d_split))
        pool.close()
        pool.join()

        return result


def centre(data, kluster, classe):
    """ fonction qui calcule les centroides.

        Args:
            data : le jeu de donnees
            kluster : tabeau des classes pour chaque observayion
            classes : tableau de 0 a k-1

        Returns:
            tableau: moyenne des observations pour chaque classe
    """
    centre_ = []
    for i in classe:
        centre_.append(data.loc[kluster==i].mean())
    return centre_

# Fonction variance
def var_poids(data, kluster, classe, means):
    """ fonction qui calcule les poids de chaque observation et la variance.

        Args:
            data : le jeu de donnees
            kluster : tabeau des classes pour chaque observayion
            classes : tableau de 0 a k-1
            means :  le moyenne des centroides

        Returns:
            poid: poid de chaque variable
            var: la moyenne des variances pour chaque groupe
    """

    poid = []
    var_ = []
    var = []
    length = []
    data_ = data.copy().reset_index(drop=True)

    # pour calculer la variance
    for i in classe:
        temp = data_.iloc[kluster==i]
        temp = temp.reset_index(drop=True)
        length.append(temp.shape[0])
        for j in range(temp.shape[0]):
            var_.append(dist(temp.iloc[j], means[i]))
        var.append(np.sum(var_))
        var_ = []

    means = pd.DataFrame(means).mean()

    # pour calculer la distance entre la moyenne total et les obeservations
    for j in range(data.shape[0]):
        poid.append(dist(data.iloc[j], means))

    var = np.array(var)
    length = np.array(length)
    poid = np.array(poid)
    poid = poid/np.sum(poid)
    var = np.sum(var/length)
    return poid, var



def kmeans(data, kluster, proc):
    """ fonction qui classifie notre jeu de donnees

        Args:
            data : le jeu de donnees
            kluster : nombre de classe
            proc : le nombrer de processeur pour la parallelisation

        Returns:
            k0: classe final pour chaque observation
    """

    # Condition d'arrêt
    noConv = True
    limit = 0
    # Standardisé les données
    data_t = pd.DataFrame()
    for i in data.columns.values.tolist():
        x = np.array(data[i])
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        data_t[i] = x

    # Indice des centroide  initials
    i_centroide = np.random.choice(data.shape[0], kluster, replace=False)
    centroide = data_t.iloc[i_centroide]
    k0 = paral_data(data_t, proc, centroide, range(kluster))
    centroide = centre(data_t, k0, range(kluster))

    while noConv and limit < 10:
        k = paral_data(data_t, proc, centroide, range(kluster))

        if (k0 == k).all():
            noConv = False
        else:
            k0 = k
            centroide = centre(data_t, k0, range(kluster))

        limit += 1

    return k0


def kmeans_up(data, kluster, proc, rep):
    """ fonction qui classifie notre jeu de donnees (version ameliorer)

        Args:
            data : le jeu de donnees
            kluster : nombre de classe
            proc : le nombrer de processeur pour la parallelisation
            rep : nombre de fois qu'on fait le  reechantillonnage

        Returns:
            lesVar: tableau des moyennes des variance pour chaque groupe pour chaque iteration
            lesKuls : tableau des classification  pour chaque iteration
    """
    # Standardiser les données
    data_t = pd.DataFrame()
    for i in data.columns.values.tolist():
        x = np.array(data[i])
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        data_t[i] = x

    lesKluster = []
    lesVar = []

    for i in range(rep):
        # Indice des centroide  initials
        i_centroide = np.random.choice(data.shape[0], kluster, replace=False)
        centroide = data_t.iloc[i_centroide]
        k0 = paral_data(data_t, proc, centroide, range(kluster))
        centroide = centre(data_t, k0, range(kluster))

        # echantillonage en se basant sur les poids
        poid = var_poids(data_t, k0, range(kluster), centroide)[0]
        i_centroide = np.random.choice(data.shape[0], kluster, replace=False, p=poid)
        centroide = data_t.iloc[i_centroide]
        k0 = paral_data(data_t, proc, centroide, range(kluster))
        centroide = centre(data_t, k0, range(kluster))

        # condition d'arret
        noConv = True
        limit = 0

        while noConv and limit < 8:
            k = paral_data(data_t, proc, centroide, range(kluster))
            if (k0 == k).all():
                noConv = False
            else:
                k0 = k
                centroide = centre(data_t, k0, range(kluster))
            limit += 1

        lesKluster.append(k0)
        lesVar.append(var_poids(data_t, k0, range(kluster), centroide)[1])

    return lesVar, lesKluster










###############################################################################

###  Amélioration avec la recherche à voisinages variables (RVV)   ###

###############################################################################

# Calculer la distance euclidienne d'une observation par rapport à un centroide
def eucludian_dist(row,centroid):
    return sqrt(sum((row-centroid)**2))


# Mettre à jour les coordonnées des centroides par le calcul des moyennes de chaque variable pour chaque groupe
def centroids_update(df_new,features_col,n_cluster): 
    
    df_new_centroid=pd.DataFrame()
    
    # Calcul de la moyenne de chaque variable pour chaque groupe
    for clust in range(n_cluster): 
        
        for col in features_col: 
            df_new_centroid.loc[clust,col]=df_new.loc[df_new.cluster==clust,col].mean()
        df_new_centroid.loc[clust,'cluster']=clust
    
    df_new_centroid['cluster']=df_new_centroid['cluster'].map(int)
    df_new_centroid.dropna(axis=0,how='any',inplace=True)
    df_new_centroid.reset_index(inplace=True)
    df_new_centroid.drop(columns='index',inplace=True)
    return df_new_centroid
            

# Affecter une observation à un groupe en fonction de la distance minimale
def assign_class(df,df_centroid,features_col):
    df_new_columns=pd.DataFrame()
    
    for i in range(len(df)):
        distances=[]
        clusters=[]
        
        # Calcul des distances par rapport aux centroides pour chaque observation
        for j in range(len(df_centroid)):
        
            row=np.array(df.loc[i,features_col])
            centroid=np.array(df_centroid.loc[j,features_col])
            dist=eucludian_dist(row,centroid)
            clust=df_centroid.loc[j,'cluster']
            distances.append(dist)
            clusters.append(clust)
        
        # Identificaion de la distance minimale et affectation d'une observation à un groupe
        df_new_columns.loc[i,'distance_min']=min(distances)
        df_new_columns.loc[i,'cluster']=clusters[distances.index(min(distances))]

    df_new_columns['cluster']=df_new_columns['cluster'].map(int)
    df_new_columns['distance_min']=df_new_columns['distance_min'].map('{:.3f}'.format)
    df_new=pd.concat([df,df_new_columns],axis=1)
    return df_new


# Vérifier la stabilité des centroides : Retourner VRAI si au moins l'un des centroides est modifié
# Si la distance entre le nouveau et l'ancien centroides est supérieure à epsilon (=1e-6) alors le centroide a changé
def centroid_change(df_centroid,df_new_centroid,features_col,epsilon):
    
    # Calcul de distance entre le nouveau et l'ancien centroides pour chaque centroide
    if (len(df_centroid)==len(df_new_centroid)) and list(df_centroid.columns)==list(df_new_centroid.columns):
        df_gap=(df_new_centroid[features_col]-df_centroid[features_col])**2
        difference=[]
        for i in range(len(df_gap)):
            gap_value=sqrt(np.array(df_gap.loc[i,:]).sum())
            difference.append(gap_value)
        
        # Comparaison de chaque distance à epsilon   
        if (np.array(difference)>epsilon).sum()>0:
            centroids_changes=True
        else:
            centroids_changes=False
        
        return centroids_changes
    else: 
        print ('Error: Check dimensions and columns of dataframes')
        return None
    

# Fonction qui met en oeuvre le k_means de base et retourne les coordonnées des centroides ainsi que
# les observations affectées avec le groupe auquel elles sont affectées
def kmeans(df,df_centroid,features_col,n_cluster,max_iter,epsilon):
    iter=0
    change=True
    while((iter<=max_iter) & (change!=False)):
        
        # Affectation des observations aux différents groupes
        df_new=assign_class(df,df_centroid,features_col)
      
        # Mise à jour des centroides
        df_new_centroid=centroids_update(df_new,features_col,n_cluster)
        
        change=centroid_change(df_centroid,df_new_centroid,features_col,epsilon)
        df_centroid=df_new_centroid
        iter+=1
    return df_new,df_new_centroid


#Fonction qui met en oeuvre la perturbation PERTMERGE et retourne les deux groupes perturbés et leurs centroides
# ainsi que les autres groupes non modifiés et leurs centroides également
def pertmerge(dataframe,df_centroid,n_cluster,features_col,percent=0.5): #dataframe=df_new
    
    # Choix aléatoire d'un groupe à perturber 
    cluster1=randint(0,n_cluster-1)
   
    df1=dataframe.loc[dataframe.cluster==cluster1,features_col].copy()
    
    # Choix aléatoire d'une observation appartenant au groupe à perturber
    idx1=np.random.randint(0,len(df1)-1)
    
    # Calcul des distances de l'observation sélectionnée avec les centroides des autres groupes
     
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
    
    # Identification du groupe le plus proche de l'observation à l'aide de la distance minimale
    dict={'cluster':cluster,'dist':distance}
    
    cluster_distance=list(dict.values())
    
    closest_cluster=cluster_distance[0][cluster_distance[1].index(min(cluster_distance[1]))]
    
    df2=dataframe[dataframe.cluster==closest_cluster].copy()
    
    # Selection et changement de groupe pour certaines observations des deux groupes voisins
    n1=round(percent*len(df1))
    n2=round(percent*len(df2))
    
    list_elements_1=list(sample(list(range(len(df1))),n1))
    list_elements_2=list(sample(list(range(len(df2))),n2))
    
    df1_cluster=dataframe.loc[dataframe.cluster==cluster1,features_col+['cluster']].copy()
    df2_cluster=dataframe.loc[dataframe.cluster==closest_cluster,features_col+['cluster']].copy()
    
    
    df1_cluster.iloc[list_elements_1,-1]=int(closest_cluster)
    df2_cluster.iloc[list_elements_2,-1]=int(cluster1)

    # Séparation des autres groupes des deux groupes voisins qui ont été modifiés
    df_otherClusters=dataframe[(dataframe.cluster!=cluster1) & (dataframe.cluster!=closest_cluster)].copy()
    df_2clusters=pd.concat([df1_cluster,df2_cluster],axis=0,ignore_index=False).sort_index()
    
    # Calcul des centroides des deux groupes voisins modifiés
    df_centroid_otherClusters=df_centroid[(df_centroid.cluster!=cluster1) & (df_centroid.cluster!=closest_cluster)].copy()
    df_centroid_2clusters=df_centroid[(df_centroid.cluster==cluster1) | (df_centroid.cluster==closest_cluster)].copy()  
    df_centroid_2clusters=centroids_update(df_2clusters,df_centroid_2clusters,n_cluster)
   
    return df_otherClusters,df_2clusters,df_centroid_otherClusters,df_centroid_2clusters
        




############################################################################################################
### Boucle pour identifier le nombre optimal de groupe en roulant k-means pour différents nombres de groupes
### et en mesurant la performance par le score silouette 
############################################################################################################

# epsilon=1e-6
# max_iter=300
# features_col=list(df.columns)
# max_cluster=20
# S_score_max=-1
# df_n_centroid=pd.DataFrame()
# best_n_cluster=None
# cluster_score=[]
# df_clusters_centroids=pd.DataFrame()

# for n_cluster in range(2,max_cluster+1):
    

#     clusters_array=np.array([i for i in range(n_cluster)])
#     centroid_index=list(sample(list(range(len(df))),n_cluster))
    
#     df_sample=df.iloc[centroid_index,:]
#     df_sample.reset_index(inplace=True)
#     df_centroid=df_sample.iloc[:,1:].copy()
#     df_centroid['cluster']=clusters_array
   
#     # Algorithme k-means
#     start_time=time.time()
#     df_new,df_new_centroid=kmeans(df,df_centroid,features_col,n_cluster,max_iter,epsilon)
#     end_time= time.time()
#     exec_time=end_time-start_time

#     # Score silhouette
#     S_score = silhouette_score(X=df_new[features_col], labels=df_new.iloc[:,-1])
#     print(n_cluster,' : ',S_score)
    
    
    
#     cluster_score.append({'n_cluster':n_cluster,'silhouette':S_score,'exec_time':exec_time})
#     df_new_centroid['num_cluster']=n_cluster
#     df_clusters_centroids=pd.concat([df_clusters_centroids,df_new_centroid],axis=0,ignore_index=True)
    


# df_cluster_score=pd.DataFrame.from_dict(cluster_score)
# df_cluster_score.sort_values(by='silhouette',axis=0,ascending=False,ignore_index=True,inplace=True)
# print(df_cluster_score)





          
############################################################################################################
### Boucle pour rouler 100 itérations de RVV sur les 5 meilleurs résultats de regroupement 
### issus du k-means de base
############################################################################################################

#topN=5
# kmax=100

# df_topN_clusters_centroids=pd.DataFrame()
# df_topN_clusters=pd.DataFrame()
# k_cluster_score=[]
# for n_cluster in list(df_cluster_score.loc[:(topN-1),'n_cluster']):
    
#     S_score_ref=df_cluster_score.loc[df_cluster_score.n_cluster==n_cluster,'silhouette'].values[0]
#     S_score_max=S_score_ref
#     duration=df_cluster_score.loc[df_cluster_score.n_cluster==n_cluster,'exec_time'].values[0]
#     k_cluster_score.append({'num_cluster':n_cluster,'k':0,'silhouette':S_score_ref,'duration':duration})
    
#     df_new_centroid=df_clusters_centroids[df_clusters_centroids.num_cluster==n_cluster].copy()
#     df_new_centroid.reset_index(inplace=True)
#     df_new_centroid.drop(columns=['index','num_cluster'],inplace=True)
#     df_new=assign_class(df,df_new_centroid,features_col)
#     df_best_cluster=df_new.copy()
#     df_best_centroid=df_new_centroid.copy()
    

#     print("REFERENCE using only initial classic Kmeans: For n_clusters =",
#           n_cluster,"The average silhouette_score is :",S_score_ref)

#     k=1
#     while k<=kmax:

#         # Choix aléatoire du pourcentage d'observations à échanger entre les deux groupes voisins à perturber       
#         percent=sample(list(np.linspace(0.5,0.9,9)),1)[0]
        
#         # Algorithme pertmerge comme perturbation
#         start=time.time()
#         df_otherClusters,df_2clusters,df_centroid_otherClusters,df_centroid_2clusters=pertmerge(dataframe=df_new,
#                                                                                             df_centroid=df_new_centroid,
#                                                                                             n_cluster=n_cluster,
#                                                                                             features_col=features_col,
#                                                                                             percent=percent)
    
    
#         df_2clusters.drop(columns=['cluster'],inplace=True)
#         df_2clusters.reset_index(inplace=True)
    
#         df_centroid_2clusters.reset_index(inplace=True)
#         df_centroid_2clusters.drop(columns=['index'],inplace=True)
    
    
#         # Algorithme k-means comme recherche locale
#         df_new_2clusters,df_new_centroid_2clusters=kmeans(df_2clusters,df_centroid_2clusters,
#                                                        features_col,n_cluster,max_iter,epsilon)
#         end=time.time()
#         duration=end-start
    
#         df_new_2clusters.set_index('index',inplace=True)
    
    
#         df_joined=pd.concat([df_otherClusters,df_new_2clusters],axis=0,ignore_index=False).sort_index()
   
#         df_centroid_joined=pd.concat([df_centroid_otherClusters,df_new_centroid_2clusters],axis=0,ignore_index=True)
    
#         S_score = silhouette_score(X=df_joined[features_col], labels=df_joined.iloc[:,-1])

    
    
#         df_new=df_joined.copy()
#         df_new_centroid=df_centroid_joined.copy()
    
#         if S_score>S_score_max:
#             S_score_max=S_score
#             df_best_centroid=df_new_centroid.copy()
#             df_best_cluster=df_new.copy()
        
#         k_cluster_score.append({'num_cluster':n_cluster,'k':k,'silhouette':S_score,'duration':duration})
#         k+=1
    
#     df_best_centroid['num_cluster']=n_cluster
#     df_best_cluster['num_cluster']=n_cluster
    
    
#     df_topN_clusters_centroids=pd.concat([df_topN_clusters_centroids,df_best_centroid],axis=0,ignore_index=True)
#     df_topN_clusters=pd.concat([df_topN_clusters,df_best_cluster],axis=0,ignore_index=True)
    
#     print("The best silhouette_score is :",S_score_max,' with a number of clusters of: ',n_cluster)
#     print("\n") 

# df_k_cluster_score=pd.DataFrame.from_dict(k_cluster_score)






if __name__ == '__main__':
    pass
