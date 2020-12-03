import numpy as np

#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    '''
    ###############################################
    # TODO: implement the Kmeans++ initialization
    ###############################################
    
    centers = []
    centers.append(generator.randint(0, n))
    
    def l2_norm_sq(a, b):
        return np.power(np.linalg.norm(a-b, axis=1), 2)
    
        
    for k in range(1, n_cluster):
        min_dist = []
        for xn in x:
            d = np.min(l2_norm_sq(x[centers], xn))
            min_dist.append(d)
        min_dist = np.array(min_dist)
        prob = min_dist / np.sum(min_dist)
        
        x_idx = np.array([i for i in range(0, len(x))])
        r = generator.rand()
#         new_center = generator.choice(x_idx, size=1, p=prob)[0]
        new_center = np.argmax(np.cumsum(prob) > r)
        centers.append(new_center)
    
    

    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)



class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array, 
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0), 
                  - number of times you update the assignment, an Int (at most self.max_iter)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        ###################################################################
        # TODO: Update means and membership until convergence 
        #   (i.e., average K-mean objective changes less than self.e)
        #   or until you have made self.max_iter updates.
        ###################################################################
        def l2_norm_sq(a, b):
            return np.power(np.linalg.norm(a-b, axis=1), 2)
        
        def distortion_objective(centroids, x, membership):
            
            J = 0
            for i in range(0, self.n_cluster):
                if(len(x[membership == i]) == 0):
                    continue
                
                J += np.sum(l2_norm_sq(x[membership == i], np.expand_dims(centroids[i], axis=0)))
            return J/len(x)
        
        def update_membership(x, centroids):
            
            c = np.expand_dims(centroids, axis=1)
            dist = np.linalg.norm(x-c, axis=2)
            membership = np.argmin(dist, axis=0)
                
            return membership
        
        def update_centroids(x, membership, centroids):
            centroids_new = []
            for i in range(0, self.n_cluster):
                centroids_new.append(np.mean(x[membership == i], axis=0))
            centroids_new = np.stack(centroids_new)
            centroids_new[np.isnan(centroids_new)] = centroids[np.isnan(centroids_new)]
            return centroids_new
            

        centroids = []
        for c in self.centers:
            centroids.append(x[c])
        centroids = np.stack(centroids)
        
        membership = np.array([0 for i in range(0, N)])
        current_J = distortion_objective(centroids, x, membership)
        i = 1
        
        while i <= self.max_iter:
            membership = update_membership(x, centroids)
            
            new_J = distortion_objective(centroids, x, membership)
            if abs(current_J - new_J) > self.e:
                current_J = new_J
                centroids = update_centroids(x, membership, centroids)                
                i+=1
            else:
                break
            
        return centroids, membership, i

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        ################################################################
        # TODO:
        # - assign means to centroids (use KMeans class you implemented, 
        #      and "fit" with the given "centroid_func" function)
        # - assign labels to centroid_labels
        ################################################################
        
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, _ = kmeans.fit(x, centroid_func=centroid_func)
        
        centroid_labels = []
        
        for i in range(0, self.n_cluster):
            labels, counts = np.unique(y[membership == i], return_counts=True)
            label = max(zip(labels, counts), key=lambda x: x[1])[0]
            centroid_labels.append(label)
        centroid_labels = np.array(centroid_labels)
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        ##########################################################################
        # TODO:
        # - for each example in x, predict its label using 1-NN on the stored 
        #    dataset (self.centroids, self.centroid_labels)
        ##########################################################################
        
        pred = np.argmin(np.linalg.norm(np.expand_dims(self.centroids, axis=1)-x, axis=2), axis=0)
        
        return self.centroid_labels[pred]




def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    ##############################################################################
    # TODO
    # - replace each pixel (a 3-dimensional point) by its nearest code vector
    ##############################################################################
    
    original_shape = image.shape
    image = np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2]))
    pred = np.argmin(np.linalg.norm(np.expand_dims(code_vectors, axis=1)-image, axis=2), axis=0)
    return code_vectors[pred].reshape(original_shape)

    
    
