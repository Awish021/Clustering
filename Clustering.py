import os
from sklearn.metrics import euclidean_distances
import pylab
import cv2
import numpy as np
import cPickle as pickle
import shutil
from sklearn.mixture import GaussianMixture
import time
from GMMInference import GMMInference

def create_objects(path):
    objs =[]
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".p"):
                obj=pickle.load(open(os.path.join(root, name),"rb"))
                obj.data=np.asarray(obj.data)
                objs.append(obj)
    return objs

def create_data_for_clustering(objs):
    data=[]
    for obj in objs:
        data=np.concatenate((data,obj.data))

    data=data.reshape(len(objs),obj.data.shape[0])
    return data

def create_clustering_folders(clusters,path):
    tmp = path
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    length = range(clusters)
    path = tmp
    for i in length:
        newdir = path + str(i)
        if not os.path.exists(newdir):
            os.makedirs(newdir)


#Regular cluster with K-Means.
# Input - k - number of clusters
#       - mat - the data matrix
#       - objs - the data objects.
# Output - folder that contains folder per cluster with all the pictures that have the same label
# Warning: The path is hard-coded.
def clusterWithKMeans(k, mat, objs):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    mat=np.float32(mat)
    compactness,labels,centers = cv2.kmeans(mat, k, None, criteria, 100, flags)
    path = r'/home/avishay/Project/Clustering-K-Means/'
    create_clustering_folders(k,path)

    for i in range(len(objs)):
        obj=objs[i]
        lb = labels[i][0]
        target=path+str(lb)+"/"+str(obj.probe_fk)+"_"+str(obj.mb_pk)+".png"
        shutil.copyfile(obj.url,target)
        #comute logliklihood
        clusters=[]
    for i in range(k):
        currentClu = mat[np.where(labels==i)[0],:]
        clusters.append(currentClu)
    return bic(np.asarray(clusters),centers)


def _loglikelihood(num_points, num_dims, clusters, centroids):
        ll = 0
        for cluster in clusters:
            fRn = len(cluster)
            t1 = fRn * np.log(fRn)
            t2 = fRn * np.log(num_points)
            variance = _cluster_variance(num_points, clusters, centroids) or np.nextafter(0, 1)
            t3 = ((fRn * num_dims) / 2.0) * np.log((2.0 * np.pi) * variance)
            t4 = (fRn - 1.0) / 2.0
            ll += t1 - t2 - t3 - t4
        return ll


def _cluster_variance( num_points, clusters, centroids):
    s = 0
    denom = float(num_points - len(centroids))
    for cluster, centroid in zip(clusters, centroids):
        distances = euclidean_distances(cluster, centroid)
        s += (distances * distances).sum()
    return s / denom

def _free_params(num_clusters, num_dims):
    return num_clusters * (num_dims + 1)

def bic(clusters, centroids):
    num_points = sum(len(cluster) for cluster in clusters)
    num_dims = clusters[0][0].shape[0]

    log_likelihood = _loglikelihood(num_points, num_dims, clusters, centroids)
    num_params = _free_params(len(clusters), num_dims)
    print log_likelihood
    return -(log_likelihood - num_params / 2.0 * np.log(num_points))

#Regular cluster with GMM-EM.
# Input - k - number of clusters
#       - mat - the data matrix
#       - objs - the data objects.
# Output - folder that contains folder per cluster with all the pictures that have the same label
# Warning: The path is hard-coded.
def clusterWithGMM(k, mat, objs):
    model = GaussianMixture(n_components=k,
                                covariance_type='full', reg_covar=0.111199)
    model.fit(mat)
    labels = model.predict(mat)

    path = r'/home/avishay/Project/Clustering-GMM-EM/'
    create_clustering_folders(k,path)

    for i in range(len(objs)):
        obj=objs[i]
        lb = labels[i]
        target=path+str(lb)+"/"+str(obj.probe_fk)+"_"+str(obj.mb_pk)+".png"
       # print target

        shutil.copyfile(obj.url,target)





def compute_GMM(mat,N, covariance_type='full'):
    models = [None for n in N]
    for i in range(len(N)):
        models[i] = GaussianMixture(n_components=N[i],max_iter = 400,
                        covariance_type=covariance_type,reg_covar=0.1101999)
        models[i].fit(mat)
    return models

# N = np.arange(1, 110)
def compute_models(mat,N):
    models = compute_GMM(mat,N)

    AIC = [m.aic(mat) for m in models]
#    BIC = [m.bic(mat) for m in models]

    i_best = np.argmin(AIC)
    gmm_best = models[i_best]
    print "best fit converged:", gmm_best.converged_
    print "AIC: n_components =  %i" % N[i_best]

def cluster_per_brand_with_GMM(objs):
    brands = []
    brand_clustering_path = r'/home/avishay/Project/ClusterPerBrand-GMM-EM/'
    if os.path.exists(brand_clustering_path):
        shutil.rmtree(brand_clustering_path)
    os.makedirs(brand_clustering_path)
    mat = create_data_for_clustering(objs)
    for obj in objs:
        brands.extend([obj.product_name])

    unique_brands = list(set(brands))
    for brand in unique_brands:
        indices = [i for i, x in enumerate(brands) if x == brand]
        mat_per_brand = mat[indices]
        N= np.arange(1,len(indices)/2+2)
        models = compute_GMM(mat_per_brand,N)
        AIC = [m.aic(mat_per_brand) for m in models]

        i_best = np.argmin(AIC)
        gmm_best = models[i_best]
        cluster_brand_inference_GMM([objs[x] for x in indices], mat_per_brand, i_best + 1, gmm_best, brand, brand_clustering_path)





# Prints the best AIC for GMM on given mat
# Wrapper function for cluster_brand_inference_GMM.
# Input: the objects.
# Output: folder that contains folder per cluster with all the pictures that have the same label
# Warning: The path is hard-coded.
# Note : K is calculated by BIC.
def cluster_brand_inference_GMM(objs, mat_per_brand, i_best, gmm_best, brand, brand_clustering_path):

    newPath = brand_clustering_path + brand +"/"
    os.makedirs(newPath)
    create_clustering_folders(i_best,newPath)
    labels = gmm_best.predict(mat_per_brand)
    #move images to their cluster's folder.
    for i in range(len(objs)):
        obj=objs[i]
        lb = labels[i]
        target=newPath+"/"+str(lb)+"/"+str(obj.probe_fk)+"_"+str(obj.mb_pk)+".png"
       # print target
        shutil.copyfile(obj.url,target)





# Wrapper function for cluster_brand_inference_Gibbs_Sampling.
# Input: objects and K.
# Output: folder that contains folder per cluster with all the pictures that have the same label
# Warning: The path is hard-coded.
def cluster_per_brand_with_Gibbs_Sampling(objs,K):
    brands = []
    brand_clustering_path = r'/home/avishay/Project/ClusterPerBrand-Gibbs_Sampling/'
    if os.path.exists(brand_clustering_path):
        shutil.rmtree(brand_clustering_path)
    os.makedirs(brand_clustering_path)
    mat = create_data_for_clustering(objs)
    for obj in objs:
        brands.extend([obj.product_name])

    unique_brands = list(set(brands))
    for brand in unique_brands:
        indices = [i for i, x in enumerate(brands) if x == brand]
        mat_per_brand = mat[indices]
        cluster_brand_inference_Gibbs_Sampling([objs[x] for x in indices],
                                               mat_per_brand,brand,K, brand_clustering_path)
def cluster_brand_inference_Gibbs_Sampling(objs, mat_per_brand, brand,K, brand_clustering_path):

    newPath = brand_clustering_path + brand +"/"
    os.makedirs(newPath)
    create_clustering_folders(K,newPath)
    labels = get_labels_from_gibbs_sampling(K, mat_per_brand)
    #move images to their cluster's folder.
    for i in range(len(objs)):
        obj=objs[i]
        lb = labels[i]
        target=newPath+"/"+str(lb)+"/"+str(obj.probe_fk)+"_"+str(obj.mb_pk)+".png"
       # print target
        shutil.copyfile(obj.url,target)




def get_labels_from_gibbs_sampling(K, mat):
    ####################Oren's Gibbs sampler################
    # data
    x = mat.T
    data_dim = x.shape[0]
    N = x.shape[1]
    hyperparams = dict()
    hyperparams['Psi'] = np.eye(data_dim) * (0.01) ** 2
    hyperparams['m'] = np.ones(data_dim) * .5  # the image is in [0,1], so take the mid value.
    hyperparams['kappa'] = 1
    hyperparams['nu'] = 100
    hyperparams['alphas'] = np.ones(K) * 100
    colors = np.random.rand(K, 3)
    gmm_inference = GMMInference(data_dim=data_dim, K=K, hyperparams=hyperparams, colors=colors)
    tic = time.clock()
    gmm_inference.gibbs_sampling(x=x, nIters=35, N=N, tf_savefig=False)
    toc = time.clock()
    print 'time:', toc - tic
    labels = gmm_inference.z_est
    return labels

def clusterWithKGibbsSampling(K, mat, objs):
    labels = get_labels_from_gibbs_sampling(K, mat)
    path = r'/home/avishay/Project/Clustering-GibbsSampling/'
    create_clustering_folders(K,path)
    for i in range(len(objs)):
        obj=objs[i]
        lb = labels[i]
        target=path+str(lb)+"/"+str(obj.probe_fk)+"_"+str(obj.mb_pk)+".png"
        shutil.copyfile(obj.url,target)

# HARD CODED PATH !
objs = create_objects('/home/avishay/Project/pickle-files66')

mat=create_data_for_clustering(objs)

X = mat
clusterWithGMM(60,mat,objs)
clusterWithKMeans(60,mat,objs)
cluster_per_brand_with_GMM(objs)

# As said - gibbs sampling does not work with dim-66
objs = create_objects('/home/avishay/Project/pickle-files15')
mat=create_data_for_clustering(objs)

cluster_per_brand_with_Gibbs_Sampling(objs,12)
clusterWithKGibbsSampling(50,mat,objs)

print 'Done'
#cluster_per_brand_with_Bayesian_GMM(objs)