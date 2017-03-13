from yann.special.gan import gan 
from yann.network import network
from yann.utils.graph import draw_network

from theano import tensor as T 
import numpy
import theano 
import cPickle
rng = numpy.random
    
class igan (object):
    """
    This class creates and train two networks a GAN and a MLP. 
    """
    def __init__ (self, init_dataset, root = '.', temperature = 3, verbose = 1):
        """
        Args:
            dataset: As usual.   
            temperature: For the softmax layer.                     
        """
        self.base_dataset = init_dataset
        f = open(self.base_dataset + '/data_params.pkl', 'rb')
        data_params = cPickle.load(f)
        f.close()        
        self.data_splits = data_params ['splits']
        self.temperature = temperature        
        if self.data_splits['p'] == 0:
            self.base_num_classes = len( self.data_splits ['base'] )
        else:
            self.base_num_classes = len( self.data_splits ['shot'] + self.data_splits ['base'] )

    def setup_gan ( self, 
                    dataset = None, 
                    params = None, 
                    cook = True, 
                    root = '.', verbose = 1 ):
        """
        This function is a demo example of a generative adversarial network. 
        This is an example code. You should study this code rather than merely run it.  

        Args: 
            dataset: Supply a dataset.    
            root: location to save down stuff. 
            params: Initialize network with parameters. 
            cook: <True> If False, won't cook.          
            verbose: Similar to the rest of the dataset.

        Returns:
            net: A Network object.

        Notes:
            This is not setup properly therefore does not learn at the moment. This network here mimics
            Ian Goodfellow's original code and implementation for MNIST adapted from his source code:
            https://github.com/goodfeli/adversarial/blob/master/mnist.yaml .It might not be a perfect 
            replicaiton, but I tried as best as I could.

        """
        if dataset is None:
            dataset = self.base_dataset

        if verbose >=2:
            print (".. Creating a GAN network")

        input_params = None

        optimizer_params =  {        
                    "momentum_type"       : 'polyak',             
                    "momentum_params"     : (0.5, 0.7, 20),      
                    "regularization"      : (0.000, 0.000),       
                    "optimizer_type"      : 'rmsprop',                
                    "id"                  : "main"
                            }


        dataset_params  = {
                                "dataset"   : dataset,
                                "type"      : 'xy',
                                "id"        : 'data'
                        }

        visualizer_params = {
                        "root"       : root + '/visualizer/gan',
                        "frequency"  : 1,
                        "sample_size": 225,
                        "rgb_filters": False,
                        "debug_functions" : False,
                        "debug_layers": True,  
                        "id"         : 'main'
                            }  
                        
        resultor_params    =    {
                    "root"      : root + "/resultor/gan",
                    "id"        : "resultor"
                                }     

        # intitialize the network
        self.gan_net = gan (      borrow = True,
                        verbose = verbose )                       
        
        self.gan_net.add_module ( type = 'datastream', 
                        params = dataset_params,
                        verbose = verbose )    
        
        self.gan_net.add_module ( type = 'visualizer',
                        params = visualizer_params,
                        verbose = verbose 
                        ) 

        self.gan_net.add_module ( type = 'resultor',
                        params = resultor_params,
                        verbose = verbose 
                        ) 
        self.mini_batch_size = self.gan_net.datastream['data'].mini_batch_size
        
        #z - latent space created by random layer
        self.gan_net.add_layer(type = 'random',
                            id = 'z',
                            num_neurons = (self.mini_batch_size,10), 
                            distribution = 'gaussian',
                            mu = 0,
                            sigma = 1,
                            # limits = (0,1),
                            verbose = verbose)
        
        #x - inputs come from dataset 1 X 784\
        self.gan_net.add_layer ( type = "input",
                        id = "x",
                        verbose = verbose, 
                        datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                    # the time. 
                        mean_subtract = False )

        # Generator layers
        if not params is None:
            input_params = params ['G1']

        self.gan_net.add_layer ( type = "dot_product",
                        origin = "z",
                        id = "G1",
                        num_neurons = 1200,
                        activation = 'relu',
                        # batch_norm   = True,
                        input_params = input_params,
                        verbose = verbose
                        ) 

        if not params is None:
            input_params = params ['G2']

        self.gan_net.add_layer ( type = "dot_product",
                        origin = "G1",
                        id = "G2",
                        num_neurons = 1200,
                        activation = 'relu',
                        # batch_norm   = True,
                        input_params = input_params,
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['G(z)']

        self.gan_net.add_layer ( type = "dot_product",
                        origin = "G2",
                        id = "G(z)",
                        num_neurons = 784,
                        activation = 'tanh',
                        input_params = input_params,
                        verbose = verbose
                        )  # This layer is the one that creates the images.
            
        #D(x) - Contains params theta_d creates features 1 X 800. 
        # Discriminator Layers
        self.gan_net.add_layer ( type = "unflatten",
                        origin = "G(z)",
                        id = "G(z)-unflattened",
                        shape = (28,28),
                        verbose = verbose )

        if not params is None:
            input_params = params ['D1-x']
        self.gan_net.add_layer ( type = "dot_product",
                        id = "D1-x",
                        origin = "x",
                        num_neurons = 1200,
                        activation = ('maxout','maxout',5),
                        regularize = True,  
                        # batch_norm  = True,
                        # dropout_rate = 0.5,    
                        input_params = input_params,                                                   
                        verbose = verbose
                        )

        self.gan_net.add_layer ( type = "dot_product",
                        id = "D1-z",
                        origin = "G(z)-unflattened",
                        input_params = self.gan_net.dropout_layers["D1-x"].params, 
                        num_neurons = 1200,
                        activation = ('maxout','maxout',5),
                        regularize = True,
                        # batch_norm  = True,
                        # dropout_rate = 0.5,                       
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['D2-x']
        self.gan_net.add_layer ( type = "dot_product",
                        id = "D2-x",
                        origin = "D1-x",
                        num_neurons = 1200,
                        activation = ('maxout','maxout',5),
                        regularize = True,       
                        # batch_norm  = True,
                        # dropout_rate = 0.5,     
                        input_params = input_params,                                                                    
                        verbose = verbose
                        )

        self.gan_net.add_layer ( type = "dot_product",
                        id = "D2-z",
                        origin = "D1-z",
                        input_params = self.gan_net.dropout_layers["D2-x"].params, 
                        num_neurons = 1200,
                        activation = ('maxout','maxout',5),
                        regularize = True,
                        # dropout_rate = 0.5,          
                        # batch_norm  = True,                    
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['D(x)']
        #C(D(x)) - This is the opposite of C(D(G(z))), real
        self.gan_net.add_layer ( type = "dot_product",
                        id = "D(x)",
                        origin = "D2-x",
                        num_neurons = 1,
                        activation = 'sigmoid',
                        verbose = verbose
                        )

        #C(D(G(z))) fake - the classifier for fake/real that always predicts fake 
        self.gan_net.add_layer ( type = "dot_product",
                        id = "D(G(z))",
                        origin = "D2-z",
                        num_neurons = 1,
                        activation = 'sigmoid',
                        input_params = self.gan_net.dropout_layers["D(x)"].params,                   
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['softmax']        
        #C(D(x)) - This is the opposite of C(D(G(z))), real
        self.gan_net.add_layer ( type = "classifier",
                        id = "softmax",
                        origin = "D2-x",
                        num_classes = self.base_num_classes,
                        activation = 'softmax',
                        verbose = verbose
                    )
        
        # objective layers 
        # discriminator objective 
        self.gan_net.add_layer (type = "tensor",
                        input =  - 0.5 * T.mean(T.log(self.gan_net.layers['D(x)'].output)) - \
                                    0.5 * T.mean(T.log(1-self.gan_net.layers['D(G(z))'].output)),
                        input_shape = (1,),
                        id = "discriminator_task",
                        verbose = verbose                        
                        )

        self.gan_net.add_layer ( type = "objective",
                        id = "discriminator_obj",
                        origin = "discriminator_task",
                        layer_type = 'value',
                        objective = self.gan_net.dropout_layers['discriminator_task'].output,
                        datastream_origin = 'data', 
                        verbose = verbose
                        )
        #generator objective 
        self.gan_net.add_layer (type = "tensor",
                        input =  - 0.5 * T.mean(T.log(self.gan_net.layers['D(G(z))'].output)),
                        input_shape = (1,),
                        id = "objective_task",
                        verbose = verbose
                        )
        self.gan_net.add_layer ( type = "objective",
                        id = "generator_obj",
                        layer_type = 'value',
                        origin = "objective_task",
                        objective = self.gan_net.dropout_layers['objective_task'].output,
                        datastream_origin = 'data', 
                        verbose = verbose
                        )   
        
        #softmax objective.    
        self.gan_net.add_layer ( type = "objective",
                        id = "classifier_obj",
                        origin = "softmax",
                        objective = "nll",
                        layer_type = 'discriminator',
                        datastream_origin = 'data', 
                        verbose = verbose
                        )
        
        # from yann.utils.graph import draw_network
        # draw_network(self.gan_net.graph, filename = 'gan.png')    
        # self.gan_net.pretty_print()
        
        if cook is True:
            self.gan_net.cook (  objective_layers = ["classifier_obj", "discriminator_obj", \
                                                                                "generator_obj"],
                                optimizer_params = optimizer_params,
                                discriminator_layers = ["D1-x","D2-x"],
                                generator_layers = ["G1","G2","G(z)"], 
                                classifier_layers = ["D1-x","D2-x","softmax"],                                                
                                softmax_layer = "softmax",
                                game_layers = ("D(x)", "D(G(z))"),
                                verbose = verbose )                      

    def train_init_gan (self, lr = (0.04, 0.001), save_after_epochs = 1, epochs= (15), verbose = 2):
        """
        This method will train the initial GAN on base dataset. 

        Args:
            lr : leanring rates to train with. Default is (0.04, 0.001) 
            epochs: Epochs to train with. Default is (15)
            save_after_epochs: Saves the network down after so many epochs.
            verbose : As usual.
        """  
 
        if verbose >=2 :
            print ( ".. Training GAN ")  

        self.gan_net.train( epochs = epochs, 
                k = 1, 
                learning_rates = lr,
                pre_train_discriminator = 0,
                validate_after_epochs = 10,
                visualize_after_epochs = 2,
                save_after_epochs = save_after_epochs,
                training_accuracy = True,
                show_progress = True,
                early_terminate = True,
                verbose = verbose)

    def setup_base_mlp (self, 
                        dataset = None, 
                        root = '.', 
                        params = None, 
                        cook = True,
                        verbose = 1 ):
        """
        This method is the same as the  tutorial on building a two layer multi-layer neural
        network. The built network is mnist->800->800->10 .It optimizes with polyak momentum and 
        rmsprop. 

        Args:
            root: save location for data
            params: Initialize network with params.
            cook: <True> If False, won't cook.                      
            dataset: an already created dataset.
        """        
        if verbose >=2:
            print (".. Creating the MLP network") 

        if dataset is None:
            dataset = self.base_dataset 
            
        input_params = None

        optimizer_params =  {        
                    "momentum_type"       : 'polyak',             
                    "momentum_params"     : (0.65, 0.9, 30),      
                    "regularization"      : (0.0001, 0.0001),       
                    "optimizer_type"      : 'rmsprop',                
                    "id"                  : "optim-base"
                            }

        dataset_params  = {
                                "dataset"   : dataset,
                                "svm"       : False, 
                                "n_classes" : self.base_num_classes,
                                "id"        : 'data-base'
                        }

        visualizer_params = {
                        "root"       : root + '/visualizer/base-network',
                        "frequency"  : 1,
                        "sample_size": 225,
                        "rgb_filters": False,
                        "debug_functions" : False,
                        "debug_layers": False,  
                        "id"         : 'visualizer-base'
                            }                          

        resultor_params    =    {
                    "root"      : root + "/resultor/base-network",
                    "id"        : "resultor-base"
                                }     

        self.base = network(   borrow = True,
                        verbose = verbose )                       
        
        self.base.add_module ( type = 'optimizer',
                        params = optimizer_params, 
                        verbose = verbose )

        self.base.add_module ( type = 'datastream', 
                        params = dataset_params,
                        verbose = verbose )

        self.base.add_module ( type = 'visualizer',
                        params = visualizer_params,
                        verbose = verbose 
                        ) 

        self.base.add_module ( type = 'resultor',
                        params = resultor_params,
                        verbose = verbose 
                        ) 

        
        self.base.add_layer ( type = "input",
                        id = "input",
                        verbose = verbose, 
                        datastream_origin = 'data-base')
        
        if not params is None:
            input_params = params ['c1']

        self.base.add_layer ( type = "conv_pool",
                        id = "c1",
                        origin = "input",
                        num_neurons = 20,
                        filter_size = (5,5),
                        pool_size = (2,2),
                        activation = 'relu',
                        regularize = True,  
                        batch_norm= True,  
                        input_params = input_params,                                                    
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['c2']
        self.base.add_layer ( type = "conv_pool",
                        id = "c2",
                        origin = "c1",
                        num_neurons = 50,
                        filter_shape = (3,3),
                        pool_size = (2,2),
                        batch_norm= True,
                        regularize = True,                                                         
                        activation = 'relu',   
                        input_params = input_params,                     
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['fc1']
        self.base.add_layer ( type = "dot_product",
                        origin = "c2",
                        id = "fc1",
                        num_neurons = 800,
                        activation = 'relu',
                        batch_norm= True,
                        regularize = True,
                        dropout_rate = 0.5,     
                        input_params = input_params,                   
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['fc2']
        self.base.add_layer ( type = "dot_product",
                        origin = "fc1",
                        id = "fc2",
                        num_neurons = 800,                    
                        activation = 'relu',
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,   
                        input_params = input_params,                     
                        verbose = verbose
                        ) 
        
        if not params is None:
            input_params = params ['softmax']
        self.base.add_layer ( type = "classifier",
                        id = "softmax",
                        origin = "fc2",
                        num_classes = self.base_num_classes,
                        activation = 'softmax',
                        regularize = True,   
                        input_params = input_params,                     
                        verbose = verbose
                        )                  

        self.base.add_layer ( type = "objective",
                        id = "obj-base",
                        origin = "softmax",
                        verbose = verbose
                        )

        # self.base.pretty_print()
        # draw_network(self.gan_net.graph, filename = 'base.png')    
        if cook is True:
            self.base.cook( optimizer = 'optim-base',
                    objective_layers = ['obj-base'],
                    datastream = 'data-base',
                    classifier = 'softmax-base',
                    verbose = verbose
                    )

    def train_base_mlp (self, 
                        lr = (0.05, 0.01, 0.001), 
                        epochs = (20, 20), 
                        save_after_epochs = 1, 
                        verbose = 2):
        """
        This method will train the initial MLP on base dataset. 

        Args:
            lr : leanring rates to train with. Default is (0.05, 0.01, 0.001)
            save_after_epochs: Saves the network down after so many epochs.            
            epochs: Epochs to train with. Default is (20, 20)        
            verbose : As usual.
        """     
        if verbose >=2 :
            print ( ".. Training Base MLP ")

        self.base.train( epochs = epochs, 
                validate_after_epochs = 10,
                visualize_after_epochs = 10,  
                save_after_epochs = save_after_epochs,                          
                training_accuracy = True,
                show_progress = True,
                early_terminate = True,
                learning_rates = lr,               
                verbose = verbose)

        self.base.test(verbose = verbose)
        
    def setup_baseline_inc(self, dataset, root = '.', verbose= 2):
        """
        This method updates the increment the mlp on the increment batch.

        Args:
            root: location to save outputs.
            dataset: Increment dataset.

        Notes:
            This network does not share parameters, from the base_mlp, but creates a new copy 
            of all the parameters with a new network.
        """
        if verbose >=2:
            print (".. Creating the increment network") 
        
        f = open(dataset + '/data_params.pkl', 'rb')
        data_params = cPickle.load(f)
        f.close()        
        self.data_splits = data_params ['splits']
        self.inc_num_classes = len(self.data_splits ['shot']) + len( self.data_splits ['base'] )
        optimizer_params =  {        
                    "momentum_type"       : 'polyak',             
                    "momentum_params"     : (0.65, 0.9, 30),      
                    "regularization"      : (0.0001, 0.0001),       
                    "optimizer_type"      : 'rmsprop',                
                    "id"                  : "optim-inc-baseline"
                            }

        dataset_params  = {
                                "dataset"   :  dataset,
                                "svm"       :  False, 
                                "n_classes" : self.inc_num_classes,
                                "id"        : 'data-inc-baseline'
                        }     

        self.baseline = network ()  

        visualizer_params = {
                        "root"       : root + '/visualizer/baseline-inc',
                        "frequency"  : 1,
                        "sample_size": 225,
                        "rgb_filters": False,
                        "debug_functions" : False,
                        "debug_layers": False,  
                        "id"         : 'visualizer-inc-baseline'
                            }     

        resultor_params    =    {
                    "root"      : root + "/resultor/baseline-inc",
                    "id"        : "resultor-inc-baseline"
                                }     

        self.baseline.add_module ( type = 'datastream', 
                        params = dataset_params,
                        verbose = verbose )

        self.baseline.add_module ( type = 'optimizer',
                        params = optimizer_params, 
                        verbose = verbose )

        self.baseline.add_module ( type = 'visualizer',
                        params = visualizer_params,
                        verbose = verbose 
                        ) 

        self.baseline.add_module ( type = 'resultor',
                        params = resultor_params,
                        verbose = verbose 
                        ) 

        self.baseline.add_layer ( type = "input",
                        id = "input",
                        verbose = verbose, 
                        datastream_origin = 'data-inc-baseline')

        base_params = self.base.get_params(verbose = verbose)                
        from yann.utils.pickle import shared_params
        base_params = shared_params (base_params)

        self.baseline.add_layer ( type = "conv_pool",
                        id = "c1",
                        origin = "input",
                        num_neurons = 20,
                        filter_size = (5,5),
                        pool_size = (2,2),
                        activation = 'relu',
                        regularize = True,  
                        batch_norm= True, 
                        input_params = base_params ['c1'],                                                                              
                        verbose = verbose
                        )

        self.baseline.add_layer ( type = "conv_pool",
                        id = "c2",
                        origin = "c1",
                        num_neurons = 50,
                        filter_shape = (3,3),
                        pool_size = (2,2),
                        batch_norm= True,
                        regularize = True,                                                         
                        activation = 'relu',       
                        input_params = base_params ['c2'],                                          
                        verbose = verbose
                        )

        self.baseline.add_layer ( type = "dot_product",
                        origin = "c2",
                        id = "fc1",
                        num_neurons = 800,
                        activation = 'relu',
                        input_params = base_params ['fc1'],        
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,                                        
                        verbose = verbose
                        )

        self.baseline.add_layer ( type = "dot_product",
                        origin = "fc1",
                        id = "fc2",
                        num_neurons = 800,                    
                        activation = 'relu',
                        input_params = base_params ['fc2'],    
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,                                                                    
                        verbose = verbose
                        ) 
        
        # For classifier layer, recreating...
        old_w = self.base.dropout_layers['softmax'].w.get_value(borrow = True)
        old_b = self.base.dropout_layers['softmax'].b.get_value(borrow = True)

        new_w = numpy.asarray(0.01 * rng.standard_normal( size=(old_w.shape[0], 
                                                               len( self.data_splits ['base'])
                                                               )
                                                               ),
                                   dtype=theano.config.floatX)
        new_w_values = numpy.concatenate((old_w,new_w), axis = 1)                                    
        new_b = numpy.asarray(0.01 * rng.standard_normal( size = (len( self.data_splits ['base']))), 
                                                        dtype=theano.config.floatX)
        new_b_values = numpy.concatenate((old_b,new_b), axis = 0)                                    
        new_w = theano.shared(value= new_w_values, name='inc-weights', borrow = True) 
        new_b = theano.shared(value= new_b_values, name='inc-bias',    borrow = True)  
      
        # This removes the last two parameters added (Which should be the softmax)

        self.baseline.add_layer ( type = "classifier",
                        id = "softmax-inc-baseline",
                        origin = "fc2",
                        num_classes = self.inc_num_classes,
                        activation = 'softmax',
                        regularize = True,                        
                        input_params = [new_w, new_b],
                        verbose = verbose
                        )

        self.baseline.add_layer ( type = "objective",
                        id = "obj-inc-baseline",
                        origin = "softmax-inc-baseline",
                        verbose = verbose
                        )

        # self.baseline.pretty_print()
        # draw_network(self.baseline.graph, filename = 'baseline.png')    
        
        self.baseline.cook( optimizer = 'optim-inc-baseline',
                objective_layers = ['obj-inc-baseline'],
                datastream = 'data-inc-baseline',
                classifier_layer = 'softmax-inc-baseline',
                verbose = verbose
                )

    def train_baseline_inc (self,
                            save_after_epochs = 1,
                            lr = (0.05, 0.01, 0.001),
                            epochs = (20, 20), 
                            verbose = 2):
        
        """
        This method will train the incremental MLP on incremental dataset. 

        Args:
            lr : leanring rates to train with. Default is (0.05, 0.01, 0.001)
            epochs: Epochs to train with. Default is (20, 20)            
            verbose : As usual.
        """   
        if verbose >= 2:
            print ".. Training baseline network"

        self.baseline.train( epochs = epochs, 
                validate_after_epochs = 1,
                visualize_after_epochs = 10,
                save_after_epochs = save_after_epochs,
                training_accuracy = True,
                show_progress = True,
                early_terminate = True,
                learning_rates = lr,               
                verbose = verbose)

        self.baseline.test(verbose = verbose)

    def setup_mentor(self, temperature = None, verbose= 2):
        """
        This method sets up the metor network which is basically the same network that takes the 
        GAN as input and produces softmaxes.
        """
        if verbose >=2:
            print (".. Creating the mentor network n") 

        self.mentor = network ()  

        if not temperature is None:
            self.temperature = temperature
            
        self.mentor.add_layer ( type = "tensor",
                        id = "input",
                        input = self.gan_net.dropout_layers['G(z)'].output,
                        input_shape = (self.mini_batch_size,784),
                        verbose = verbose )

        self.mentor.add_layer ( type = "unflatten",
                        id = "input-unflattened",
                        origin ="input",
                        shape = (28,28),
                        verbose = verbose
                        )

        self.mentor.add_layer ( type = "conv_pool",
                        id = "c1",
                        origin = "input-unflattened",
                        num_neurons = 20,
                        filter_size = (5,5),
                        pool_size = (2,2),
                        activation = 'relu',
                        regularize = True,  
                        batch_norm= True, 
                        input_params = self.base.dropout_layers['c1'].params,                                                                                                  
                        verbose = verbose
                        )

        self.mentor.add_layer ( type = "conv_pool",
                        id = "c2",
                        origin = "c1",
                        num_neurons = 50,
                        filter_shape = (3,3),
                        pool_size = (2,2),
                        batch_norm= True,
                        regularize = True,                                                         
                        activation = 'relu',       
                        input_params = self.base.dropout_layers['c2'].params,                                                                                                  
                        verbose = verbose
                        )

        
        self.mentor.add_layer ( type = "dot_product",
                        origin = "c2",
                        id = "fc1",
                        num_neurons = 800,
                        activation = 'relu',
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,            
                        input_params = self.base.dropout_layers['fc1'].params,                                                                                                                                      
                        verbose = verbose
                        )

        self.mentor.add_layer ( type = "dot_product",
                        origin = "fc1",
                        id = "fc2",
                        num_neurons = 800,                    
                        activation = 'relu',
                        input_params = self.base.dropout_layers['fc2'].params,     
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,                                           
                        verbose = verbose
                        ) 
        
        self.mentor.add_layer ( type = "classifier",
                        id = "softmax",
                        origin = "fc2",
                        num_classes = self.base.dropout_layers['softmax'].output_shape[1],
                        activation = 'softmax',
                        input_params = self.base.dropout_layers['softmax'].params,         
                        regularize = True,                                       
                        verbose = verbose
                        )

        self.mentor.add_layer ( type = "classifier",
                        id = "softmax-base-temperature",
                        origin = "fc2",
                        num_classes = self.base.dropout_layers['softmax'].output_shape[1],
                        activation = ('softmax', self.temperature),
                        input_params = self.base.dropout_layers['softmax'].params,  
                        regularize = True,                                                                      
                        verbose = verbose
                        )                        

        # self.mentor.pretty_print()
        # draw_network(self.mentor.graph, filename = 'mentor.png')    
        

    def setup_hallucinated_inc(self, dataset, root = '.', verbose= 2):
        """
        This method setup the increment the mlp on the increment net.

        Args:
            root: location to save outputs.        
            dataset: Increment dataset.

        Notes:
            This method creates two networks with shared parameters. One network is used to update
            the parameters using the dataset and the other network is used to update the parameters
            for the mentoring via GAN.

            The parameters are not shared with the mentor network they are newly created copies, but
            the two networks created in this method do share parameters.
        """
        if verbose >=2:
            print (".. Creating the increment network with mentoring") 
        
        f = open(dataset + '/data_params.pkl', 'rb')
        data_params = cPickle.load(f)
        f.close()        
        self.data_splits = data_params ['splits']
        optimizer_params =  {
                    "momentum_type"       : 'polyak',             
                    "momentum_params"     : (0.65, 0.9, 30),      
                    "regularization"      : (0.0001, 0.0001),       
                    "optimizer_type"      : 'rmsprop',                
                    "id"                  : "optim-inc-hallucinated"
                            }

        dataset_params  = {
                                "dataset"   :  dataset,
                                "svm"       :  False, 
                                "n_classes" :  len (self.data_splits['base']) ,
                                "id"        : 'data-inc-hallucinated'
                        }     

        visualizer_params = {
                        "root"       : root + '/visualizer/hallucinated-inc',
                        "frequency"  : 1,
                        "sample_size": 225,
                        "rgb_filters": False,
                        "debug_functions" : False,
                        "debug_layers": False,  
                        "id"         : 'hallucinated'
                            }     

        resultor_params    =    {
                    "root"      : root + "/resultor/hallucianted-inc",
                    "id"        : "hallucinated"
                                }     

        self.hallucinated = network()   

        self.hallucinated.add_module ( type = 'datastream', 
                        params = dataset_params,
                        verbose = verbose )

        self.hallucinated.add_module ( type = 'optimizer',
                        params = optimizer_params, 
                        verbose = verbose )

        self.hallucinated.add_module ( type = 'visualizer',
                        params = visualizer_params,
                        verbose = verbose 
                        ) 

        self.hallucinated.add_module ( type = 'resultor',
                        params = resultor_params,
                        verbose = verbose 
                        ) 

        # Collecting parameters as copies from the base mentor network.
        base_params = self.base.get_params(verbose = verbose)                
        from yann.utils.pickle import shared_params
        base_params = shared_params (base_params)

        ##########
        # Network from dataset just the incremental network inference.
        ##########
        self.hallucinated.add_layer ( type = "input",
                        id = "data",
                        verbose = verbose, 
                        datastream_origin = 'data-inc-hallucinated')

        self.hallucinated.add_layer ( type = "conv_pool",
                        id = "c1-data",
                        origin = "data",
                        num_neurons = 20,
                        filter_size = (5,5),
                        pool_size = (2,2),
                        activation = 'relu',
                        regularize = True,  
                        batch_norm= True, 
                        input_params = base_params ['c1'],                                                                              
                        verbose = verbose
                        )

        self.hallucinated.add_layer ( type = "conv_pool",
                        id = "c2-data",
                        origin = "c1-data",
                        num_neurons = 50,
                        filter_shape = (3,3),
                        pool_size = (2,2),
                        batch_norm= True,
                        regularize = True,                                                         
                        activation = 'relu',       
                        input_params = base_params ['c2'],                                          
                        verbose = verbose
                        )

        self.hallucinated.add_layer ( type = "dot_product",
                        origin = "c2-data",
                        id = "fc1-data",
                        num_neurons = 800,
                        activation = 'relu',
                        input_params = base_params ['fc1'],     
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,                                           
                        verbose = verbose )

        self.hallucinated.add_layer ( type = "dot_product",
                        origin = "fc1-data",
                        id = "fc2-data",
                        num_neurons = 800,                    
                        activation = 'relu',
                        input_params = base_params ['fc2'],     
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,                                                                   
                        verbose = verbose ) 

        # For classifier layer, recreating...   
    
        old_w = base_params ['softmax'][0].eval()
        old_b = base_params ['softmax'][1].eval()

        if self.inc_num_classes > old_w.shape[1]:
            assert len(self.data_splits ['shot']) == old_w.shape[1]
            
            new_w = numpy.asarray(0.01 * rng.standard_normal( size=(old_w.shape[0], 
                                                                len( self.data_splits ['base'])
                                                                )
                                                                ),
                                    dtype=theano.config.floatX)
            new_w_values = numpy.concatenate((old_w,new_w), axis = 1)   

            new_b = numpy.asarray(0.01 * rng.standard_normal( size = (len( 
                                                                self.data_splits ['base']))), 
                                                           dtype=theano.config.floatX)
            new_b_values = numpy.concatenate((old_b,new_b), axis = 0)  
        else:
            assert self.inc_num_classes == old_w.shape[1]
            
            new_w_values = old_w
            new_b_values = old_b 

        new_w = theano.shared(value= new_w_values, name='inc-weights', borrow = True) 
        new_b = theano.shared(value= new_b_values, name='inc-bias',    borrow = True)  
      
        # This removes the last two parameters added (Which should be the softmax)
        # This works on the labels from the dataset.
        self.hallucinated.add_layer ( type = "classifier",
                        id = "softmax-inc-hallucinated-data",
                        origin = "fc2-data",
                        num_classes = self.inc_num_classes,
                        activation = 'softmax',
                        input_params = [new_w, new_b],
                        regularize = True,                        
                        verbose = verbose  )
            

        ##########
        # Softmax temperature of incremental network from GAN inputs.
        ##########
        
        self.hallucinated.add_layer ( type = "tensor",
                        id = "gan-input",
                        input = self.gan_net.inference_layers [ 'G(z)'].output,
                        input_shape = self.gan_net.dropout_layers ['G(z)'].output_shape,
                        verbose = verbose )

        self.hallucinated.add_layer ( type = "unflatten",
                        id = "gan-input-unflattened",
                        origin ="gan-input",
                        shape = (28,28),
                        verbose = verbose
                        )

        self.hallucinated.add_layer ( type = "conv_pool",
                        id = "c1-gan",
                        origin = "gan-input-unflattened",
                        num_neurons = 20,
                        filter_size = (5,5),
                        pool_size = (2,2),
                        activation = 'relu',
                        regularize = True,  
                        batch_norm= True,       
                        input_params = self.hallucinated.dropout_layers ['c1-data'].params,                                                                    
                        verbose = verbose
                        )

        self.hallucinated.add_layer ( type = "conv_pool",
                        id = "c2-gan",
                        origin = "c1-gan",
                        num_neurons = 50,
                        filter_shape = (3,3),
                        pool_size = (2,2),
                        batch_norm= True,
                        regularize = True,                                                         
                        activation = 'relu',    
                        input_params = self.hallucinated.dropout_layers ['c2-data'].params,                                                                                                                
                        verbose = verbose
                        )

        self.hallucinated.add_layer ( type = "dot_product",
                        origin = "c2-gan",
                        id = "fc1-gan",
                        num_neurons = 800,
                        activation = 'relu',
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,                        
                        input_params = self.hallucinated.dropout_layers['fc1-data'].params,                        
                        verbose = verbose )

        self.hallucinated.add_layer ( type = "dot_product",
                        origin = "fc1-gan",
                        id = "fc2-gan",
                        num_neurons = 800,                    
                        activation = 'relu',
                        batch_norm= True,
                        dropout_rate = 0.5,
                        regularize = True,                        
                        input_params = self.hallucinated.dropout_layers['fc2-data'].params,                                                  
                        verbose = verbose  )                         
    
        self.hallucinated.add_layer ( type = "classifier",
                        id = "softmax-inc-hallucinated-gan",
                        origin = "fc2-gan",
                        num_classes = self.inc_num_classes,
                        activation = ('softmax', self.temperature),
                        input_params = self.hallucinated.dropout_layers \
                                                           ['softmax-inc-hallucinated-data'].params,      
                        regularize = True,                                                                       
                        verbose = verbose )  

        ##########
        # This will make the mentor values available to the current network so that we 
        # can caluclate errors
        ##########
        if self.inc_num_classes > old_w.shape[1]:
            # Needed only if mentor and this has different number of classes.
            self.hallucinated.add_layer(type = "random",
                                                id = 'zero-targets',
                                                num_neurons = (self.mini_batch_size, \
                                                                self.inc_num_classes - \
                                                                                old_w.shape[1] ), 
                                                distribution = 'binomial',
                                                p = 0,
                                                verbose = verbose)

            input_shape = [self.mentor.layers['softmax-base-temperature'].output_shape,
                        self.hallucinated.layers['zero-targets'].output_shape]                       

        # importing a layer from the mentor network. 
        self.hallucinated.add_layer (type = "tensor",
                                            id = 'merge-import',
                                            input = self.mentor.inference_layers \
                                                            ['softmax-base-temperature'].output,
                                            input_shape = self.mentor.inference_layers \
                                                    ['softmax-base-temperature'].output_shape,
                                            verbose = verbose )

        if self.inc_num_classes > old_w.shape[1]:
            # This layer is a 10 node output of the softmax temperature. Sets up the mentor targets.
            self.hallucinated.add_layer (type = "merge",
                                                layer_type = "concatenate",
                                                id = "mentor-target",
                                                origin = ( 'merge-import', 'zero-targets' ),
                                                verbose = verbose
                                                )
            mentor_target = 'mentor-target'
        else:
            mentor_target = 'merge-import'

        ##########        
        # objective layers
        ##########
        # This is the regular classifier objective for the incremental net.
        self.hallucinated.add_layer ( type = "objective",
                        id = "obj-inc",
                        origin = "softmax-inc-hallucinated-data",
                        verbose = verbose
                        )

        # This is error between the temperature softmax layer and the mentor target.
        # This provides the incremental update.
        self.hallucinated.add_layer (type = "merge",
                        id = "obj-temperature",
                        layer_type = "error",
                        error = "rmse",
                        origin = ("softmax-inc-hallucinated-gan", mentor_target),
                         )      

        # self.hallucinated.pretty_print()
        # draw_network(self.hallucinated.graph, filename = 'hallucinated.png')    
        
        self.hallucinated.cook( optimizer = 'optim-inc-hallucinated',
                objective_layers = ['obj-inc','obj-temperature'],
                objective_weights = [1, 1],
                datastream = 'data-inc-hallucinated',
                classifier_layer = 'softmax-inc-hallucinated-data',
                verbose = verbose
                )

    def train_hallucinated_inc (self, 
                                save_after_epochs = 1,
                                lr = (0.05, 0.01, 0.001), 
                                epochs = (20, 20), 
                                verbose = 2):
        """
        This method will train the incremental MLP on incremental dataset. 

        Args:
            lr : leanring rates to train with. Default is (0.05, 0.01, 0.001)
            epochs: Epochs to train with. Default is (20, 20)               
            verbose : As usual.
        """     

        if verbose >=2 :
            print (".. Training hallucinated network")                  
        self.hallucinated.train( epochs = epochs, 
                validate_after_epochs = 1,
                visualize_after_epochs = 10,   
                save_after_epochs = save_after_epochs,             
                training_accuracy = True,
                show_progress = True,
                early_terminate = True,
                learning_rates = lr,               
                verbose = verbose)

        self.hallucinated.test(verbose = verbose)


if __name__ == '__main__':
    pass