import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    
    assert len(real_labels) == len(predicted_labels)
    predicted_labels = np.array(predicted_labels)
    real_labels = np.array(real_labels)
    TP = np.dot(real_labels, predicted_labels).sum()
    FP = ((predicted_labels == 1.0) & (real_labels == 0.0)).sum()
    FN = ((predicted_labels == 0.0) & (real_labels == 1.0)).sum()
    
    if(TP+FP == 0.0):
        return 0.0
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    if(precision+recall == 0.0):
        return 0.0

    return float((2*precision*recall) / (precision+recall))



class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.power(np.power(np.absolute(point1-point2), 3).sum(), 1/3)


    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.power(np.power(point1-point2, 2).sum(), 1/2)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        norm = (np.linalg.norm(point1)*np.linalg.norm(point2))
        return (1.0-np.dot(point1, point2) / norm) if norm != 0 else 1.0



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        best_k = -1
        best_df = None
        best_model = None
        best_f1 = -1
        for k in range(1, 30, 2):
            
            if(k > len(y_train)):
                break
            for i, df in enumerate(distance_funcs.keys()):
                model = KNN(k, distance_funcs[df])
                model.train(x_train, y_train)
                pred = model.predict(x_val)
                f1 = f1_score(y_val, pred)
                
                if(f1 > best_f1):
                    best_k = k
                    best_df = i
                    best_model = model
                    best_f1 = f1
                elif(f1 == best_f1):
                    if(i < best_df):
                        best_k = k
                        best_df = i
                        best_model = model
                        best_f1 = f1
                    elif(i == best_df and k < best_k):
                        best_k = k
                        best_df = i
                        best_model = model
                        best_f1 = f1
                    
        print("best", best_f1)
        # You need to assign the final values to these variables
        self.best_scaler = None
        self.best_k = best_k
        self.best_distance_function = list(distance_funcs.keys())[best_df]
        self.best_model = best_model
        

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        best_k = -1
        best_df = None
        best_scaler = None
        best_model = None
        best_f1 = -1
        
        for k in range(1, 30, 2):
            if(k > len(y_train)):
                break
            for i, df in enumerate(distance_funcs.keys()):
                for j, scaler in enumerate(scaling_classes.keys()):
                    scaled_x_train = scaling_classes[scaler]()(x_train)
                    scaled_x_val = scaling_classes[scaler]()(x_val)
                    model = KNN(k, distance_funcs[df])
                    model.train(scaled_x_train, y_train)
                    pred = model.predict(scaled_x_val)
                    f1 = f1_score(y_val, pred)
                    if(f1 > best_f1):
                        best_k = k
                        best_df = i
                        best_model = model
                        best_f1 = f1
                        best_scaler = j

                    elif(f1 == best_f1):
                        if(j < best_scaler):
                            best_k = k
                            best_df = i
                            best_model = model
                            best_f1 = f1
                            best_scaler = j
                        elif(j == best_scaler and i < best_df):
                            best_k = k
                            best_df = i
                            best_model = model
                            best_f1 = f1
                            best_scaler = j
                        elif(j == best_scaler and i == best_df and k < best_k):
                            best_k = k
                            best_df = i
                            best_model = model
                            best_f1 = f1
                            best_scaler = j
                    
                    
                
        print("BEST with scaling", best_f1)
        # You need to assign the final values to these variables
        self.best_k = best_k
        self.best_scaler = list(scaling_classes.keys())[best_scaler]
        self.best_distance_function = list(distance_funcs.keys())[best_df]
        self.best_model = best_model
        


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        
        def helper(x):
            norm = float(np.linalg.norm(x))
            return x/norm if norm!=0 else x
        return np.apply_along_axis(helper, 1, features)


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features)
        def helper(x):
            norm = float(np.max(x)-np.min(x))
            return (x-np.min(x))/norm if norm!=0 else np.zeros(x.shape)
        
        return np.apply_along_axis(helper, 0, features)
