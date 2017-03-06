if __name__ == '__main__':
    
    import sys
    from incremental_gan import igan
    from dataset import cook_split_base as base_dataset
    from dataset import cook_split_inc as inc_dataset    
    # from dataset import cook_mnist_complete as base_dataset 

    base = base_dataset (verbose = 2)
    base = base.dataset_location()

    inc = inc_dataset (verbose = 2)
    inc = inc.dataset_location()    

    # This will initialize the igan. Both MLP and GAN 
    # will be training with the base dataset.
    igan_obj = igan ( base, temperature = 2, verbose = 2 )
    
    lr = (0.04, 0.005, 0.001)
    epochs =(15, 15)
    igan_obj.train_init_gan ( lr = lr, epochs = epochs, verbose = 2 )

    # For all the MLP, we use the same learning rates and epochs.
    # For optimizers, go inside the setup function and change them.
    lr = (0.05, 0.01)
    epochs =(10)
    igan_obj.train_base_mlp ( lr =lr, epochs = epochs, verbose = 2 )    

    # This will initialize the baseline incremental network.
    # This network is intended to demonstrate catastrophic forgetting
    # will be training with the increment dataset.
    igan_obj.setup_baseline_inc ( inc, verbose = 2 )
    igan_obj.train_baseline_inc ( lr =lr, epochs = epochs, verbose = 2 )

    # This will initialize and train the hallucinated incremental network.
    # This network is intended to deomonstrate that hallucinating from GAN 
    # could allow us to learn incremental learning. This is counter to 
    # what the baseline demonstrates.
    igan_obj.setup_mentor (verbose = 2)
    igan_obj.setup_hallucinated_inc ( inc, verbose = 2 )
    igan_obj.train_hallucinated_inc ( lr =lr, epochs = epochs, verbose = 2 )    