import sys, os
from svhn import cgan
from dataset import cook_continual as make_dataset

if __name__ == '__main__':
    
    gan_lr = (0.00005, 0.001)    
    gan_epochs =(30)
    mlp_lr = (0.00005, 0.00001)    
    mlp_epochs =(50)
    cook = True 

    system = cgan()

    # This is the first increment.
    initial_split = {   "train"  : [0,1,2,3],
                        "test"   : [0,1,2,3]  }  
    init_dataset = make_dataset (splits = initial_split, verbose = 1)
    dataset = init_dataset.dataset_location()   
    print ("\n\n\n                       ---------- Base MLP ------------ \
        \n\n\n ")   
    system.setup_base_mlp ( dataset = dataset, cook = cook, verbose = 2)
    system.train_mlp(mlp = system.base, lr = mlp_lr, epochs = mlp_epochs)

    print ("\n\n\n                       ---------- Base GAN ------------ \
        \n\n\n ")          
    system.create_gan(dataset = dataset, cook = cook, verbose = 2)
    system.train_gan( lr = gan_lr, epochs = gan_epochs)

    # Now that the base and the gans are setup for the 
    # first increment, this will run the second increment.    

    
    
    ########################################################################

    print ("\n\n\n                   ---------- First Increment PS ------------ \
        \n\n\n ")           

    system.update_phantom_labeler(temperature = 2, verbose = 2)

    initial_split = {   "train"  : [4,5,6],
                        "test"   : [0,1,2,3,4,5,6]  }  
    init_dataset = make_dataset (splits = initial_split, verbose = 1)
    dataset = init_dataset.dataset_location()  

    print ("\n\n\n                   ---------- First Increment MLP ------------ \
        \n\n\n ")           
    system.update_current_network(dataset = dataset, verbose = 2)    
    system.train_mlp (mlp = system.current, lr = mlp_lr, epochs = mlp_epochs)       
    system.update_base()

    print ("\n\n\n                   ---------- First Increment GAN ------------ \
        \n\n\n ")          
    system.create_gan(dataset = dataset, cook = cook, verbose = 2)
    system.train_gan( lr = gan_lr, epochs = gan_epochs)



    ########################################################################


    initial_split = {   "train"  : [7,8,9],
                        "test"   : [0,1,2,3,4,5,6,7,8,9]  }  
    init_dataset = make_dataset (splits = initial_split, verbose = 1)
    dataset = init_dataset.dataset_location()     

    print ("\n\n\n                   ---------- Second Increment PS ------------ \
        \n\n\n ")           

    system.update_phantom_labeler(temperature = 2, verbose = 2)

    print ("\n\n\n                   ---------- Second Increment MLP ------------ \
        \n\n\n ")           
    system.update_current_network(dataset = dataset, verbose = 2)    
    system.train_mlp (mlp = system.current, lr = mlp_lr, epochs = mlp_epochs)