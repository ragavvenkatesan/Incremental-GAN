import sys, os
from svhn import igan
from dataset import cook_split_base as base_dataset

if __name__ == '__main__':
      
    # from dataset import cook_mnist_complete as base_dataset 
    base_splits = { "base"              : [0,1,2,3,4,5],
                    "shot"              : [6,7,8,9],
                    "p"                 : 0    }  


    data_loc = '/home/local/ASUAD/rvenka10/Dropbox/svhn/' # this is where you will add the matlab
                                            # data files as processed from the 
                                            # yann mat2yann tutorial.
    base = base_dataset (location = data_loc,
                          splits = base_splits, 
                          verbose = 1)
    base = base.dataset_location()
    # Data was once made and stored here. Uncomment the lines above
    # instead of the one below to newly make one.
    # base = '_datasets/_dataset_88730'  

    # This will initialize the igan. Both MLP and GAN 
    # will be training with the base dataset.
    root = 'records/site_1'
    if not os.path.exists(root):
        os.makedirs(root)        
    
    # initialize igan object
    site1 = igan ( init_dataset = base, root = root, verbose = 1 )

    # setup and train site-1 Base MLP
    site1.setup_base_mlp(root = root, verbose = 1)    
    lr = (0.04, 0.001)    
    epochs =(15)
    site1.train_base_mlp ( lr =lr, 
                           save_after_epochs = 1,
                           epochs = epochs, 
                           early_terminate= False,
                           verbose = 2 )

    # setup and train site-2 GAN
    site1.setup_gan(root = root, verbose = 1)
    
    lr = (0.004, 0.00001, 0.00001)    
    epochs =(15, 15)
    site1.train_init_gan ( lr = lr, 
                           save_after_epochs = 1,
                           epochs = epochs,
                           verbose = 2 )