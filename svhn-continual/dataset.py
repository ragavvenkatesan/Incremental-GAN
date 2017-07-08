from yann.special.datasets import split_continual

def cook_continual(verbose = 1, **kwargs):
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
    location = '/home/local/ASUAD/rvenka10/Dropbox/raw_datasets/svhn/'  
    # location = '/Users/ragav/Dropbox/raw_datasets/svhn/'
    
    if not 'data_params' in kwargs.keys():
        data_params = {
                   "source"             : 'matlab',
                   # "name"               : 'yann_svhn', # some name.
                   "location"     : location,    # some location to load from.  
                   "height"             : 32,
                   "width"              : 32,
                   "channels"           : 3,
                   "batches2test"       : 13,
                   "batches2train"      : 100,
                   "batches2validate"   : 13,
                   "mini_batches_per_batch" : (10, 10, 10),                   
                   "mini_batch_size"    : 500}

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
                        "train"             : [0,1],
                        "test"              : [0,1,2,3],
                    }     
    else:
        splits = kwargs ['splits']

    dataset = split_continual(dataset_init_args = data_params,
                            save_directory = save_directory,
                            preprocess_init_args = preprocess_params,
                            split_args = splits,
                            verbose = 3)
    return dataset

if __name__ == '__main__':
    pass
