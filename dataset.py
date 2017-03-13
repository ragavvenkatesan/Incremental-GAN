from yann.utils.dataset import *
import numpy

def cook_mnist_complete(verbose = 1, **kwargs):
    """
    Wrapper to cook mnist dataset that creates the whole thing. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:

        This will also have the split parameter.
    """

    if not 'data_params' in kwargs.keys():
        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (100, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : True,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    if not 'splits' in  kwargs.keys():
        splits = { 
                        "base"              : [0,1,2,3,4,5,6,7,8,9],
                        "shot"              : [],
                        "p"                 : 0
                    }     
    else:
        splits = kwargs ['splits']

    dataset = split_only_train(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            split_args = splits,
                            verbose = 3)
    return dataset

def cook_split_base(verbose = 1, **kwargs):
    """
    Wrapper to cook mnist dataset that also creates a split dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        
        The base of this dataset will be classes 0,1,2,4,5 and the split will be classes
        6,7,8,9.
    """

    if not 'data_params' in kwargs.keys():
        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (100, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : True,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    if not 'splits' in  kwargs.keys():
        splits = { 
                        "base"              : [0,1,2,3,4,5],
                        "shot"              : [6,7,8,9],
                        "p"                 : 0
                    }     
    else:
        splits = kwargs ['splits']

    dataset = split_all(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            split_args = splits,
                            verbose = 3)
    return dataset

def cook_split_inc(verbose = 1, **kwargs):
    """
    Wrapper to cook mnist dataset that also creates the rest of the dataset. Will take as input,

    Args:

        save_directory: which directory to save the cooked dataset onto.
        dataset_parms: default is the dictionary. Refer to :mod:`setup_dataset`
        preprocess_params: default is the dictionary. Refer to :mod:`setup_dataset`

    Notes:
        
        The base of this dataset will be classes 0,1,2,4,5,7,9 and the split will be classes
        3,6,8.
    """

    if not 'data_params' in kwargs.keys():
        data_params = {
                   "source"             : 'skdata',
                   "name"               : 'mnist',
                   "location"			: '',
                   "mini_batch_size"    : 500,
                   "mini_batches_per_batch" : (100, 20, 20),
                   "batches2train"      : 1,
                   "batches2test"       : 1,
                   "batches2validate"   : 1,
                   "height"             : 28,
                   "width"              : 28,
                   "channels"           : 1  }

    else:
        data_params = kwargs['data_params']

    if not 'preprocess_params' in kwargs.keys():

    # parameters relating to preprocessing.
        preprocess_params = {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"     : True,
                        }
    else:
        preprocess_params = kwargs['preprocess_params']

    if not 'save_directory' in kwargs.keys():
        save_directory = '_datasets'
    else:
        save_directory = kwargs ['save_directory']

    if not 'splits' in  kwargs.keys():
        splits = { 
                        "base"              : [6,7,8,9],
                        "shot"              : [0,1,2,3,4,5],
                        "p"                 : 0
                    }     
    else:
        splits = kwargs ['splits']

    dataset = split_only_train(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            split_args = splits,
                            verbose = 3)
    return dataset

class split_all(setup_dataset):
    """
    Inheriting from the setup dataset. The new methods added will include the split. 
    """    
    def __init__(self,
                 dataset_init_args,
                 save_directory = '_datasets',
                 verbose = 0,
                 **kwargs): 
        """
        This is just a re-use of the setup_dataset. This will construct the split dataset. 

        Args:
            split_args: Is a dictionary of the form,
                    split_args =  { 
                            "base"              : [0,1,2,4,6,8,9],
                            "shot"              : [3,5,7],
                            "p"                 : 0
                            }   
        Notes:
            Arguments are the same as in the case of setup_dataset. With the addtion of one extra
            argument  - split_args.
        """

        if "preprocess_init_args" in kwargs.keys():
            self.preprocessor = kwargs['preprocess_init_args']
        else:
            self.preprocessor =  {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"		: True,
                            }        

        if "split_args" in kwargs.keys():
            self.splits = kwargs['split_args']
        else:
            self.splits = { 
                            "base"              : [0,1,2,3,4,5],
                            "shot"              : [6,7,8,9],
                            "p"                 : 0
                        } 

        super(split_all,self).__init__(     dataset_init_args = dataset_init_args,
                                            save_directory = save_directory,
                                            preprocess_init_args = self.preprocessor,
                                            verbose = 1)

    def _create_skdata_mnist(self, verbose = 1):
        """
        Interal function. Use this to create mnist and cifar image datasets
        This is modfied for the split dataset from the original ``setup_dataset`` class.
        """
        if verbose >=2:
            print (".. Creating a split dataset")
        if verbose >=3:
            print("... Importing " + self.name + " from skdata")
        data = getattr(thismodule, 'load_skdata_' + self.name)()

        if verbose >=2:
            print(".. setting up dataset")
            print(".. training data")
        # Assuming this is num classes. Dangerous ?
        self.n_classes = data[0][1].max()

        data_x, data_y, data_y1 = self._split_data (data[0])
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        training_sample_size = data_x.shape[0]
        training_batches_available  = int(numpy.floor(training_sample_size / self.mini_batch_size))

        if training_batches_available < self.batches2train * self.mini_batches_per_batch[0]:
            self.mini_batches_per_batch = ( training_batches_available/self.batches2train,
                                            self.mini_batches_per_batch [1],
                                            self.mini_batches_per_batch [2] )

        if self.batches2train * self.mini_batches_per_batch[0] < self.cache_images[0]:
            self.cache_images = (self.mini_batches_per_batch[0] * self.mini_batch_size, \
                                        self.cache_images[1],  self.cache_images[2])

        data_x = data_x[:self.cache_images[0]]
        data_y = data_y[:self.cache_images[0]]                

        loc = self.root + "/train/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2train):
            start_index = batch * self.cache_images[0]
            end_index = start_index + self.cache_images[0]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        if verbose >=2:
            print(".. validation data ")

        data_x, data_y, data_y1 = self._split_data (data[1])
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        validation_sample_size = data_x.shape[0]
        validation_batches_available = int(numpy.floor(
                                                validation_sample_size / self.mini_batch_size))

        if validation_batches_available < self.batches2validate * self.mini_batches_per_batch[1]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            validation_batches_available/self.batches2validate,
                                            self.mini_batches_per_batch [2] )

        if self.batches2validate * self.mini_batches_per_batch[1] < self.cache_images[1]:
            self.cache_images = (   self.cache_images[0],\
                                    self.mini_batches_per_batch[1] * self.mini_batch_size, \
                                    self.cache_images[2])

        data_x = data_x[:self.cache_images[1]]
        data_y = data_y[:self.cache_images[1]]

        loc = self.root + "/valid/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2validate):
            start_index = batch * self.cache_images[1]
            end_index = start_index + self.cache_images[1]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        if verbose >=2:
            print(".. testing data ")

        data_x, data_y, data_y1 = self._split_data(data[2])
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )
        testing_sample_size = data_x.shape[0]
        testing_batches_available = int(numpy.floor(testing_sample_size / self.mini_batch_size))

        if testing_batches_available < self.batches2test * self.mini_batches_per_batch[2]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            self.mini_batches_per_batch [1],
                                            testing_batches_available/self.batches2test )

        if self.batches2test * self.mini_batches_per_batch[2] < self.cache_images[2]:
            self.cache_images = (   self.cache_images[0],\
                                    self.cache_images[1], \
                                    self.mini_batches_per_batch[2] * self.mini_batch_size )

        data_x = data_x[:self.cache_images[2]]
        data_y = data_y[:self.cache_images[2]]

        loc = self.root + "/test/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2test):
            start_index = batch * self.cache_images[2]
            end_index = start_index + self.cache_images[2]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        dataset_args = {
                "location"                  : self.root,
                "mini_batch_size"           : self.mini_batch_size,
                "cache_batches"             : self.mini_batches_per_batch,
                "batches2train"             : self.batches2train,
                "batches2test"              : self.batches2test,
                "batches2validate"          : self.batches2validate,
                "height"                    : self.height,
                "width"                     : self.width,
                "channels"              : 1 if self.preprocessor ["grayscale"] else self.channels,
                "cache"                     : self.cache,
                "splits"                    : self.splits
                }

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()

    def _split_data (self, data):
        """
        This is an internal method that will split the datasets.

        Args:
            data: train, test and valid batches in a tuple.
        
        Returns:
            tuple: split data in the same format as data.
        """
        n_shots = self.splits["p"]
        data_x, data_y, data_y1  = data
        locs = numpy.zeros(len(data_y), dtype = bool)
        for label in xrange(self.n_classes + 1):
            temp = numpy.zeros(len(data_y), dtype = bool)                                                
            temp[data_y==label] = True
            if label in self.splits["shot"]:
                count = 0        
                for element in xrange(len(temp)):
                    if temp[element] == True:    # numpy needs == rather than 'is'               
                        count = count + 1
                    if count > n_shots:	                     	            
                        temp[element] = False	
                        		     
            locs[temp] = True
        data_x = data_x[locs]
        data_y = data_y[locs]
        data_y1 = data_y1[locs]
        return (data_x, data_y, data_y1)  

class split_only_train(setup_dataset):
    
    """
    Inheriting from the split dataset. The new methods added will include the split. 
    """    
    def __init__(self,
                 dataset_init_args,
                 save_directory = '_datasets',
                 verbose = 0,
                 **kwargs): 
        """
        This is just a re-use of the setup_dataset. This will construct the split dataset. 

        Args:
            split_args: Is a dictionary of the form,
                splits = { 
                        "base"              : [6,7,8,9],
                        "shot"              : [0,1,2,3,4,5],
                        "p"                 : 0
                    }   
        Notes:
            Arguments are the same as in the case of setup_dataset. With the addtion of one extra
            argument  - split_args.
        """
        if "preprocess_init_args" in kwargs.keys():
            self.preprocessor = kwargs['preprocess_init_args']
        else:
            self.preprocessor =  {
                            "normalize"     : True,
                            "ZCA"           : False,
                            "grayscale"     : False,
                            "zero_mean"		: True,
                            }        

        if "split_args" in kwargs.keys():
            self.splits = kwargs['split_args']
        else:
            self.splits =  { 
                        "base"              : [6,7,8,9],
                        "shot"              : [0,1,2,3,4,5],
                        "p"                 : 0
                            }   

        self.n_classes = len(self.splits['base']) + len(self.splits['shot'])
        super(split_only_train,self).__init__( 
                                        dataset_init_args = dataset_init_args,
                                        save_directory = save_directory,
                                        preprocess_init_args = self.preprocessor,
                                        verbose = 1)

    def _create_skdata_mnist(self, verbose = 1):
        """
        Interal function. Use this to create mnist and cifar image datasets
        This is modfied for the split dataset from the original ``setup_dataset`` class.
        """
        if verbose >=2:
            print (".. Creating a split dataset")
        if verbose >=3:
            print("... Importing " + self.name + " from skdata")
        data = getattr(thismodule, 'load_skdata_' + self.name)()

        if verbose >=2:
            print(".. setting up dataset")
            print(".. training data")

        data_x, data_y, data_y1 = self._split_data (data[0])

        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        training_sample_size = data_x.shape[0]
        training_batches_available  = int(numpy.floor(training_sample_size / self.mini_batch_size))

        if training_batches_available < self.batches2train * self.mini_batches_per_batch[0]:
            self.mini_batches_per_batch = ( training_batches_available/self.batches2train,
                                            self.mini_batches_per_batch [1],
                                            self.mini_batches_per_batch [2] )

        if self.batches2train * self.mini_batches_per_batch[0] < self.cache_images[0]:
            self.cache_images = (self.mini_batches_per_batch[0] * self.mini_batch_size, \
                                        self.cache_images[1],  self.cache_images[2])

        data_x = data_x[:self.cache_images[0]]
        data_y = data_y[:self.cache_images[0]]                

        loc = self.root + "/train/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2train):
            start_index = batch * self.cache_images[0]
            end_index = start_index + self.cache_images[0]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        if verbose >=2:
            print(".. validation data ")

        data_x, data_y, data_y1 = data[1]
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )

        validation_sample_size = data_x.shape[0]
        validation_batches_available = int(numpy.floor(
                                                validation_sample_size / self.mini_batch_size))

        if validation_batches_available < self.batches2validate * self.mini_batches_per_batch[1]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            validation_batches_available/self.batches2validate,
                                            self.mini_batches_per_batch [2] )

        if self.batches2validate * self.mini_batches_per_batch[1] < self.cache_images[1]:
            self.cache_images = (   self.cache_images[0],\
                                    self.mini_batches_per_batch[1] * self.mini_batch_size, \
                                    self.cache_images[2])

        data_x = data_x[:self.cache_images[1]]
        data_y = data_y[:self.cache_images[1]]

        loc = self.root + "/valid/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2validate):
            start_index = batch * self.cache_images[1]
            end_index = start_index + self.cache_images[1]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        if verbose >=2:
            print(".. testing data ")

        data_x, data_y, data_y1 = data[2]
        data_x = preprocessing ( data = data_x,
                                 height = self.height,
                                 width = self.width,
                                 channels = self.channels,
                                 args = self.preprocessor )
        testing_sample_size = data_x.shape[0]
        testing_batches_available = int(numpy.floor(testing_sample_size / self.mini_batch_size))

        if testing_batches_available < self.batches2test * self.mini_batches_per_batch[2]:
            self.mini_batches_per_batch = ( self.mini_batches_per_batch [0],
                                            self.mini_batches_per_batch [1],
                                            testing_batches_available/self.batches2test )

        if self.batches2test * self.mini_batches_per_batch[2] < self.cache_images[2]:
            self.cache_images = (   self.cache_images[0],\
                                    self.cache_images[1], \
                                    self.mini_batches_per_batch[2] * self.mini_batch_size )

        data_x = data_x[:self.cache_images[2]]
        data_y = data_y[:self.cache_images[2]]

        loc = self.root + "/test/"
        data_x = check_type(data_x, theano.config.floatX)
        data_y = check_type(data_y, theano.config.floatX)

        for batch in xrange(self.batches2test):
            start_index = batch * self.cache_images[2]
            end_index = start_index + self.cache_images[2]
            data2save = (data_x [start_index:end_index,], data_y[start_index:end_index,] )
            pickle_dataset(loc = loc, data = data2save, batch=batch)

        dataset_args = {
                "location"                  : self.root,
                "mini_batch_size"           : self.mini_batch_size,
                "cache_batches"             : self.mini_batches_per_batch,
                "batches2train"             : self.batches2train,
                "batches2test"              : self.batches2test,
                "batches2validate"          : self.batches2validate,
                "height"                    : self.height,
                "width"                     : self.width,
                "channels"              : 1 if self.preprocessor ["grayscale"] else self.channels,
                "cache"                     : self.cache,
                "splits"                    : self.splits
                }

        assert ( self.height * self.width * self.channels == numpy.prod(data_x.shape[1:]) )
        f = open(self.root +  '/data_params.pkl', 'wb')
        cPickle.dump(dataset_args, f, protocol=2)
        f.close()        

    def _split_data (self, data):
        """
        This is an internal method that will split the datasets.

        Args:
            data: train, test and valid batches in a tuple.
        
        Returns:
            tuple: split data in the same format as data.
        """
        n_shots = self.splits["p"]
        data_x, data_y, data_y1  = data
        locs = numpy.zeros(len(data_y), dtype = bool)
        for label in xrange(self.n_classes + 1):
            temp = numpy.zeros(len(data_y), dtype = bool)                                                
            temp[data_y==label] = True
            if label in self.splits["shot"]:
                count = 0        
                for element in xrange(len(temp)):
                    if temp[element] == True:    # numpy needs == rather than 'is'     
                        count = count + 1
                    if count > n_shots:	                     	            
                        temp[element] = False	
                        		     
            locs[temp] = True
        data_x = data_x[locs]
        data_y = data_y[locs]
        data_y1 = data_y1[locs]   
        return (data_x, data_y, data_y1)  

if __name__ == '__main__':
    pass