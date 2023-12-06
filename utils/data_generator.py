import numpy as np
from keras.utils import Sequence
from itertools import combinations as com

class DataGenerator(Sequence):
    
    def __init__(
            self,
            X:np.ndarray,
            y:np.ndarray,
            batch_size:int,
            augmentation=False,
            ):
        
        self.batch_size=batch_size
        self.augmentation=augmentation
        self.X = X
        self.y = y
        self.input_shape = X.shape[1:]
        self.uid_list = list(np.arange(len(X)))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.uid_list)/self.batch_size))
    
    def __getitem__(self, index):

        # Generate list of index corresponding to the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Get new list of UIDs
        new_uid_list = [self.uid_list[i] for i in indexes]

        # Generate corresponding data
        X,y = self.__data_generator(new_uid_list)
        
        return X,y
    
    def shuffle_indexes(self):
        self.indexes = np.arange(len(self.uid_list))
        np.random.shuffle(self.indexes)
    
    def on_epoch_end(self):
        self.shuffle_indexes()
    
    def __data_generator(self, uids):

        X = np.zeros(shape=(self.batch_size, *self.input_shape))
        y = np.zeros(shape=(self.batch_size))

        for i,uid in enumerate(uids):
            X[i] = self.X[uid]
            y[i] = self.y[uid]

        return X,y
    
class DataGeneratorCenters(Sequence):

    def __init__(
            self,
            X:np.ndarray,
            y:np.ndarray,
            metadata:np.ndarray,
            ):
        
        self.X = X
        self.y = y
        centers = np.unique(metadata)
        self.batch_size=len(centers)
        self.input_shape = X.shape[1:]
        self.uid_list = list(np.arange(len(X)))
        self.uid_list_C1 = []
        self.uid_list_C2 = []
        self.uid_list_C5 = []
        for i in range(len(self.uid_list)):
            if metadata[i]==centers[0]:
                self.uid_list_C1.append(i)
            elif metadata[i]==centers[1]:
                self.uid_list_C2.append(i)
            else:
                self.uid_list_C5.append(i)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.uid_list)/self.batch_size))
    
    def __getitem__(self, index):

        # Generate list of index corresponding to the batch
        uids = self.uids[index*self.batch_size:(index+1)*self.batch_size]

        # Get new list of UIDs
        new_uid_list = uids

        # Generate corresponding data
        X,y = self.__data_generator(new_uid_list)

        return X,y
    
    def shuffle_indexes(self):

        uids = np.zeros(3*np.min([len(self.uid_list_C1),len(self.uid_list_C2),len(self.uid_list_C5)]))
        self.uid_list_C1 = np.random.permutation(self.uid_list_C1)
        self.uid_list_C2 = np.random.permutation(self.uid_list_C2)
        self.uid_list_C5 = np.random.permutation(self.uid_list_C5)

        for i in range(len(uids)):
            if i%3==0:
                uids[i] = self.uid_list_C1[int(i/3)]
            elif i%3==1:
                uids[i] = self.uid_list_C2[int(i/3)]
            else:
                uids[i] = self.uid_list_C5[int(i/3)]

        self.uids = uids.astype(int)
    
    def on_epoch_end(self):
        self.shuffle_indexes()
    
    def __data_generator(self, uids):

        X = np.zeros(shape=(self.batch_size, *self.input_shape))
        y = np.zeros(shape=(self.batch_size))

        for i,uid in enumerate(uids):
            X[i] = self.X[uid]
            y[i] = self.y[uid]

        return X,y
    
class DataGeneratorInterMixUp(Sequence):

    def __init__(
            self,
            X:np.ndarray,
            y:np.ndarray,
            batch_size:int,
            p_0_1:float=0.1,
            same_class=True, # True: same class is chosen for pairs
        ):

        self.batch_size=batch_size
        self.X = X
        self.y = y
        self.same_class = same_class
        self.p_0_1 = p_0_1

        self.all_pairs = list(com(range(len(self.X)),2))
        self.n_all_pairs = len(self.all_pairs)
        self.all_pairs_0 = list(com(np.where(self.y==0)[0],2))
        self.n_all_pairs_0 = len(self.all_pairs_0)
        self.all_pairs_1 = list(com(np.where(self.y==1)[0],2))
        self.n_all_pairs_1 = len(self.all_pairs_1)

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X)/self.batch_size))
    
    def get_random_pairs(self):

        '''Generates len(self.X) pairs of indexes, assuming that each sample is seen at least once, and that each pair is unique'''


        if not self.same_class:
            indexes = np.random.choice(self.n_all_pairs, size=len(self.X), replace=False)
            self.pairs = np.array(self.all_pairs)[indexes]

        else:
            indexes_0 = np.random.choice(self.n_all_pairs_0, size=len(self.X)//2, replace=False)
            indexes_1 = np.random.choice(self.n_all_pairs_1, size=len(self.X)//2, replace=False)

            self.pairs = np.concatenate([np.array(self.all_pairs_0)[indexes_0],np.array(self.all_pairs_1)[indexes_1]], axis=0)  

    def on_epoch_end(self):
        self.get_random_pairs()
    
    def __getitem__(self, index):

        # Generate list of index corresponding to the batch
        indexes = self.pairs[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate corresponding data
        X,y = self.__data_generator(indexes)

        return X,y
    
    def __data_generator(self, indexes):

        X = np.zeros(shape=(self.batch_size, *self.X.shape[1:]))
        y = np.zeros(shape=(self.batch_size))

        for i,index in enumerate(indexes):
            if np.random.random()<self.p_0_1:
                alpha = np.random.choice([0,1])
            else:
                alpha = np.random.uniform(0,1)
            X[i] = alpha*self.X[index[0]] + (1-alpha)*self.X[index[1]]
            y[i] = alpha*self.y[index[0]] + (1-alpha)*self.y[index[1]]

        return X,y