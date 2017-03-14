import sys, os
from cifar10 import igan
from dataset import cook_split_base as base_dataset

if __name__ == '__main__':
      
    # from dataset import cook_mnist_complete as base_dataset 
    base_splits = { "base"              : [0,1,2,3,4,5],
                    "shot"              : [6,7,8,9],
                    "p"                 : 0    }  

    base = base_dataset (splits = base_splits, verbose = 1)
    base = base.dataset_location()  

    # This will initialize the igan. Both MLP and GAN 
    # will be training with the base dataset.
    root = 'records/site_1'
    if not os.path.exists(root):
        os.makedirs(root)        
    
    # initialize igan object
    site1 = igan ( init_dataset = base, root = root, verbose = 1 )

    # setup and train site-1 Base MLP
    site1.setup_base_mlp(root = root, verbose = 2)    
    lr = (0.04, 0.01, 0.0001)    
    epochs =(15, 15)
    site1.train_base_mlp ( lr =lr, 
                           save_after_epochs = 1,
                           epochs = epochs, 
                           early_terminate = False,
                           verbose = 2 )   

    # setup and train site-2 GAN
    site1.setup_gan(root = root, verbose = 1)
    lr = (0.00005, 0.001, 0.0001)    
    epochs =(20, 20)
    site1.train_init_gan ( lr = lr, 
                           save_after_epochs = 1,
                           epochs = epochs,
                           verbose = 2 )