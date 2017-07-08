from yann.special.gan import gan 
from yann.network import network
from yann.utils.graph import draw_network
from yann.utils.pickle import shared_params

from collections import OrderedDict
from theano import tensor as T 
import numpy
import theano 
import cPickle
rng = numpy.random
    
class cgan (object):
    """
    This class creates and train many incremental networks with GANs for each part and a MLP. 
    """
    def __init__ (self, init_dataset = None, root = '.', temperature = 3, verbose = 1):
        """
        Args:
            dataset: As usual.   
            temperature: For the softmax layer.    
            n_classes: Number of classes.                 
        """
        self.dataset = list()
        self.increment = 0                 
        if init_dataset is not None:
            self.dataset.append(init_dataset)
            f = open(self.dataset[self.increment] + '/data_params.pkl', 'rb')
            data_params = cPickle.load(f)
            f.close()        
        self.temperature = temperature   
        self.num_classes = 0     
        # if init_dataset is not None:          
        #    self.num_classes += len( self.data_splits ['train'] )
        self.gans = OrderedDict()

    def _gan (self,
              dataset = None,
              params = None,
              cook = True,
              root = '.', verbose = 1):
        """
        This function is a demo example of a generative adversarial network. 
        This is an example code. You should study this code rather than merely run it.  

        Args: 
            dataset: Supply a dataset.    
            root: location to save down stuff. 
            params: Initialize network with parameters. 
            cook: <True> If False, won't cook.         
            increment: which number of GAN to  
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
            dataset = self.dataset[-1]
        else:
            self.dataset.append(dataset)

        if verbose >=2:
            print (".. Creating the initial GAN network")

        input_params = None

        optimizer_params =  {        
                    "momentum_type"       : 'nesterov',             
                    "momentum_params"     : (0.65, 0.7, 15),      
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
                        "root"       : root + '/visualizer/gan_' + str(self.increment),
                        "frequency"  : 1,
                        "sample_size": 225,
                        "rgb_filters": False,
                        "debug_functions" : False,
                        "debug_layers": True,  
                        "id"         : 'main'
                            }  
                        
        resultor_params    =    {
                    "root"      : root + "/resultor/gan_" + str(self.increment),
                    "id"        : "resultor"
                                }     

        # intitialize the network
        gan_net = gan ( borrow = True, verbose = verbose )                       
        
        gan_net.add_module ( type = 'datastream', 
                        params = dataset_params,
                        verbose = verbose )    
        
        gan_net.add_module ( type = 'visualizer',
                        params = visualizer_params,
                        verbose = verbose 
                        ) 

        gan_net.add_module ( type = 'resultor',
                        params = resultor_params,
                        verbose = verbose 
                        ) 
        self.gan_mini_batch_size = gan_net.datastream['data'].mini_batch_size
        
        #z - latent space created by random layer
        gan_net.add_layer(type = 'random',
                            id = 'z',
                            num_neurons = (self.gan_mini_batch_size,10), 
                            distribution = 'gaussian',
                            mu = 0,
                            sigma = 1,
                            # limits = (0,1),
                            verbose = verbose)
        
        #x - inputs come from dataset 1 X 784\
        gan_net.add_layer ( type = "input",
                        id = "x",
                        verbose = verbose, 
                        datastream_origin = 'data', # if you didnt add a dataset module, now is 
                                                    # the time. 
                        mean_subtract = False )

        # Generator layers
        if not params is None:
            input_params = params ['G1']

        gan_net.add_layer ( type = "dot_product",
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

        gan_net.add_layer ( type = "dot_product",
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

        gan_net.add_layer ( type = "dot_product",
                        origin = "G2",
                        id = "G(z)",
                        num_neurons = 784,
                        activation = 'tanh',
                        input_params = input_params,
                        verbose = verbose
                        )  # This layer is the one that creates the images.
            
        # D(x) - Contains params theta_d creates features 1 X 800. 
        # Discriminator Layers
        gan_net.add_layer ( type = "unflatten",
                        origin = "G(z)",
                        id = "G(z)-unflattened",
                        shape = (28,28),
                        verbose = verbose )

        if not params is None:
            input_params = params ['D1-x']
        gan_net.add_layer ( type = "dot_product",
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

        gan_net.add_layer ( type = "dot_product",
                        id = "D1-z",
                        origin = "G(z)-unflattened",
                        input_params = gan_net.dropout_layers["D1-x"].params, 
                        num_neurons = 1200,
                        activation = ('maxout','maxout',5),
                        regularize = True,
                        # batch_norm  = True,
                        # dropout_rate = 0.5,                       
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['D2-x']
        gan_net.add_layer ( type = "dot_product",
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

        gan_net.add_layer ( type = "dot_product",
                        id = "D2-z",
                        origin = "D1-z",
                        input_params = gan_net.dropout_layers["D2-x"].params, 
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
        gan_net.add_layer ( type = "dot_product",
                        id = "D(x)",
                        origin = "D2-x",
                        num_neurons = 1,
                        activation = 'sigmoid',
                        verbose = verbose
                        )

        #C(D(G(z))) fake - the classifier for fake/real that always predicts fake 
        gan_net.add_layer ( type = "dot_product",
                        id = "D(G(z))",
                        origin = "D2-z",
                        num_neurons = 1,
                        activation = 'sigmoid',
                        input_params = gan_net.dropout_layers["D(x)"].params,                   
                        verbose = verbose
                        )

        if not params is None:
            input_params = params ['softmax']        
        #C(D(x)) - This is the opposite of C(D(G(z))), real
        gan_net.add_layer ( type = "classifier",
                        id = "softmax",
                        origin = "D2-x",
                        num_classes = self.num_classes,
                        activation = 'softmax',
                        verbose = verbose
                    )
        
        # objective layers 
        # discriminator objective 
        gan_net.add_layer (type = "tensor",
                        input =  - 0.5 * T.mean(T.log(gan_net.layers['D(x)'].output)) - \
                                    0.5 * T.mean(T.log(1-gan_net.layers['D(G(z))'].output)),
                        input_shape = (1,),
                        id = "discriminator_task",
                        verbose = verbose                        
                        )

        gan_net.add_layer ( type = "objective",
                        id = "discriminator_obj",
                        origin = "discriminator_task",
                        layer_type = 'value',
                        objective = gan_net.dropout_layers['discriminator_task'].output,
                        datastream_origin = 'data', 
                        verbose = verbose
                        )

        #generator objective 
        gan_net.add_layer (type = "tensor",
                        input =  - 0.5 * T.mean(T.log(gan_net.layers['D(G(z))'].output)),
                        input_shape = (1,),
                        id = "objective_task",
                        verbose = verbose
                        )

        gan_net.add_layer ( type = "objective",
                        id = "generator_obj",
                        layer_type = 'value',
                        origin = "objective_task",
                        objective = gan_net.dropout_layers['objective_task'].output,
                        datastream_origin = 'data', 
                        verbose = verbose
                        )   
        
        #softmax objective.    
        gan_net.add_layer ( type = "objective",
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
            gan_net.cook (  objective_layers = ["classifier_obj", "discriminator_obj", \
                                                                                "generator_obj"],
                                optimizer_params = optimizer_params,
                                discriminator_layers = ["D1-x","D2-x"],
                                generator_layers = ["G1","G2","G(z)"], 
                                classifier_layers = ["D1-x","D2-x","softmax"],                                                
                                softmax_layer = "softmax",
                                game_layers = ("D(x)", "D(G(z))"),
                                verbose = verbose )   
        return gan_net     

    def create_gan ( self, 
                    dataset = None, 
                    params = None, 
                    cook = True, 
                    root = '.', verbose = 1 ):
        """
        This function is a demo sets up one additional GAN on the new dataset.  

        Args: 
            dataset: Supply a dataset.    
            root: location to save down stuff. 
            params: Initialize network with parameters. 
            cook: <True> If False, won't cook.         
            increment: which number of GAN to  
            verbose: Similar to the rest of the dataset.

        Returns:
            net: A Network object.

        """
        self.dataset.append(dataset)
        f = open(self.dataset[-1] + '/data_params.pkl', 'rb')
        data_params = cPickle.load(f)
        f.close()                  
        splits = data_params["splits"]
        # This will only work if the classes are ordered 
        # from 0, .... 
        # also that max of test is basically the number of 
        # classes involved. 
        self.num_classes = max( splits ['test'] ) + 1 
        gan_net = self._gan(dataset = dataset, 
                            params = params,
                            cook = cook,
                            root = root,
                            verbose = verbose)
        self.gans[self.increment] = gan_net 
        self.increment += 1  

    def train_gan ( self, gan = None, lr = (0.04, 0.001), 
                    save_after_epochs = 1, epochs= (15), verbose = 2):
        """
        This method will train the initial GAN on base dataset. 

        Args:
            lr : leanring rates to train with. Default is (0.04, 0.001) 
            epochs: Epochs to train with. Default is (15)
            save_after_epochs: Saves the network down after so many epochs.
            verbose : As usual.
        """  

        if gan is None:
            gan = self.gans[self.increment - 1]
        if verbose >=2 :
            print ( ".. Training GAN " )  

        gan.train( epochs = epochs, 
                k = 1, 
                learning_rates = lr,
                pre_train_discriminator = 0,
                validate_after_epochs = 10,
                visualize_after_epochs = 2,
                save_after_epochs = save_after_epochs,
                training_accuracy = True,
                show_progress = True,
                early_terminate = False,
                verbose = verbose)

    def _mlp (  self, 
                id,
                dataset = None, 
                root = '.', 
                params = None, 
                num_classes = None,
                cook = True,
                verbose = 1 ):
        """
        This method is initializes and trains an MLP on some dataset.

        Args:
            root: save location for data
            params: Initialize network with params.
            cook: <True> If False, won't cook.    
            increment: which increment of MLP should be trained.    
            id: For directory setup.          
            dataset: an already created dataset.
        """        
        if verbose >=2:
            print (".. Creating the MLP network") 

        if dataset is None:
            dataset = self.base_dataset 
            
        if num_classes is None:
            num_classes = self.num_classes

        input_params = None 

        optimizer_params =  {        
                    "momentum_type"       : 'false',             
                    "momentum_params"     : (0.65, 0.9, 30),      
                    "regularization"      : (0.0001, 0.0001),       
                    "optimizer_type"      : 'adam',                
                    "id"                  : "optim"
                            }

        dataset_params  = {
                                "dataset"   : dataset,
                                "svm"       : False, 
                                "n_classes" : num_classes,
                                "id"        : 'data'
                        }

        visualizer_params = {
                        "root"       : root + '/visualizer/network-' + id,
                        "frequency"  : 1,
                        "sample_size": 225,
                        "rgb_filters": False,
                        "debug_functions" : False,
                        "debug_layers": False,  
                        "id"         : 'visualizer'
                            }                          

        resultor_params    =    {
                    "root"      : root + "/resultor/network-" + id,
                    "id"        : "resultor"
                                }     

        net = network(   borrow = True,
                        verbose = verbose )                       
        
        net.add_module ( type = 'optimizer',
                        params = optimizer_params, 
                        verbose = verbose )

        net.add_module ( type = 'datastream', 
                        params = dataset_params,
                        verbose = verbose )

        net.add_module ( type = 'visualizer',
                        params = visualizer_params,
                        verbose = verbose 
                        ) 

        net.add_module ( type = 'resultor',
                        params = resultor_params,
                        verbose = verbose 
                        ) 

        net.add_layer ( type = "input",
                        id = "input",
                        verbose = verbose, 
                        datastream_origin = 'data')
        
        if not params is None:
            input_params = params ['c1']

        net.add_layer ( type = "conv_pool",
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
        net.add_layer ( type = "conv_pool",
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
        net.add_layer ( type = "dot_product",
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
        net.add_layer ( type = "dot_product",
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
        net.add_layer ( type = "classifier",
                        id = "softmax",
                        origin = "fc2",
                        num_classes = num_classes,
                        activation = 'softmax',
                        regularize = True,   
                        input_params = input_params,                     
                        verbose = verbose
                        )                  

        net.add_layer ( type = "objective",
                        id = "obj",
                        origin = "softmax",
                        verbose = verbose
                        )

        # self.base.pretty_print()
        # draw_network(self.gan_net.graph, filename = 'base.png')    
        if cook is True:
            net.cook( optimizer = 'optim',
                    objective_layers = ['obj'],
                    datastream = 'data',
                    classifier = 'softmax',
                    verbose = verbose
                    )
        return net

    def setup_base_mlp (  self, 
                dataset = None, 
                root = '.', 
                params = None, 
                cook = True,
                verbose = 1 ):
        """
        This method is sets up the first MLP on some dataset.

        Args:
            root: save location for data
            params: Initialize network with params.
            cook: <True> If False, won't cook.    
            increment: which increment of MLP should be trained.              
            dataset: Latest created dataset.
        """ 
        if dataset is None:
            dataset = self.dataset[-1]
              
        self.base = self._mlp(dataset = dataset, 
                            params = params,
                            cook = cook,
                            root = root,
                            id = 'base' + str(self.increment-1),
                            num_classes = self.num_classes,
                            verbose = verbose)
        self.mini_batch_size = self.base.layers['input'].output_shape[0]

    def train_mlp (     self,
                        mlp,
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
            print ( ".. Training MLP ")

        mlp.train( epochs = epochs, 
                validate_after_epochs = 10,
                visualize_after_epochs = 10,  
                save_after_epochs = save_after_epochs,                          
                training_accuracy = True,
                show_progress = True,
                early_terminate = False,
                learning_rates = lr,               
                verbose = verbose)

        mlp.test(verbose = verbose)

    def update_phantom_labeler(self, temperature = 3, verbose= 2):
        """
        This method sets up the phantom labeler network which is basically the same 
        network that takes the GAN as input and produces softmaxes.
        """
        if verbose >=2:
            print (".. Creating the labeler network") 

        self.phantom_labeler = network ()  

        if not temperature is None:
            self.temperature = temperature

        # Every layer should have increment number of layers.  But the same 
        # parameters are shared. So it is in effect just one layer.            
        for inc in xrange(len(self.gans.keys())):
            self.phantom_labeler.add_layer ( type = "tensor",
                            id = "input-" + str(inc),
                            input = self.gans[inc].dropout_layers['G(z)'].output,
                            input_shape = (self.mini_batch_size,784),
                            verbose = verbose )
 
            self.phantom_labeler.add_layer ( type = "unflatten",
                            id = "input-unflattened-" + str(inc),
                            origin = "input-" + str(inc),
                            shape = (28,28),
                            verbose = verbose
                            )
        
            self.phantom_labeler.add_layer ( type = "conv_pool",
                            id = "c1-" + str(inc),
                            origin = "input-unflattened-" + str(inc),
                            num_neurons = 20,
                            filter_size = (5,5),
                            pool_size = (2,2),
                            activation = 'relu',
                            regularize = True,  
                            batch_norm= True, 
                            input_params = self.base.dropout_layers['c1'].params,                                                                                                  
                            verbose = verbose
                            )

            self.phantom_labeler.add_layer ( type = "conv_pool",
                            id = "c2-" + str(inc),
                            origin = "c1-" + str(inc),
                            num_neurons = 50,
                            filter_shape = (3,3),
                            pool_size = (2,2),
                            batch_norm= True,
                            regularize = True,                                                         
                            activation = 'relu',       
                            input_params = self.base.dropout_layers['c2'].params,                                                                                                  
                            verbose = verbose
                            )

            self.phantom_labeler.add_layer ( type = "dot_product",
                            origin = "c2-" + str(inc),
                            id = "fc1-" + str(inc),
                            num_neurons = 800,
                            activation = 'relu',
                            batch_norm= True,
                            dropout_rate = 0.5,
                            regularize = True,            
                            input_params = self.base.dropout_layers['fc1'].params,                                                                                                                                      
                            verbose = verbose
                            )

            self.phantom_labeler.add_layer ( type = "dot_product",
                            origin = "fc1-" + str(inc),
                            id = "fc2-" + str(inc),
                            num_neurons = 800,                    
                            activation = 'relu',
                            input_params = self.base.dropout_layers['fc2'].params,     
                            batch_norm= True,
                            dropout_rate = 0.5,
                            regularize = True,                                           
                            verbose = verbose
                            ) 
        
            self.phantom_labeler.add_layer ( type = "classifier",
                            id = "softmax-" + str(inc),
                            origin = "fc2-" + str(inc),
                            num_classes = self.base.dropout_layers['softmax'].output_shape[1],
                            activation = 'softmax',
                            input_params = self.base.dropout_layers['softmax'].params,         
                            regularize = True,                                       
                            verbose = verbose
                            )

            self.phantom_labeler.add_layer ( type = "classifier",
                            id = "phantom-" + str(inc),
                            origin = "fc2-" + str(inc),
                            num_classes = self.base.dropout_layers['softmax'].output_shape[1],
                            activation = ('softmax', self.temperature),
                            input_params = self.base.dropout_layers['softmax'].params,  
                            regularize = True,                                                                      
                            verbose = verbose
                            )                        

        #self.phantom_labeler.pretty_print()
        draw_network(self.phantom_labeler.graph, filename = 'phantom_labeler.png')    
        
    def update_current_network(self, dataset, root = '.', verbose= 2):
        """
        This method setup the increment net.

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

        # Collecting parameters as copies from the base mentor network.
        # New copies of params so that I wont overwrite the 
        # pointer params in the phantom sampler network object.
        params = self.base.get_params(verbose = verbose)                
        params = shared_params (params)

        # For classifier layer, recreating...   
        old_w = params ['softmax'][0].eval()
        old_b = params ['softmax'][1].eval()

        new_w = numpy.asarray(0.01 * rng.standard_normal( size=(old_w.shape[0], \
                                        self.num_classes - old_w.shape[1])),
                                        dtype=theano.config.floatX)
        new_w_values = numpy.concatenate((old_w,new_w), axis = 1)   
        new_b = numpy.asarray(0.01 * rng.standard_normal( size = \
                                            (self.num_classes - old_w.shape[1])), 
                                            dtype=theano.config.floatX)
        new_b_values = numpy.concatenate((old_b,new_b), axis = 0)  

        new_w = theano.shared(value= new_w_values, name='inc-weights', borrow = True) 
        new_b = theano.shared(value= new_b_values, name='inc-bias',    borrow = True) 

        params["softmax"] = [new_w, new_b]

        # This net is initialized with the same parameters as the phantom / base network
        # but with more nodes in the last softmax layer to accommodate for more labels.
        self.current = self._mlp(
                            dataset = dataset, 
                            params = params,
                            cook = False,
                            root = root,
                            id = str(self.increment),
                            verbose = verbose) 
        
        # AGain will have increment number of GANs. 
        objective_layers = list()
        for inc in xrange(len(self.gans.keys())):   
                          
            self.current.add_layer ( type = "tensor",
                            id = "gan-input-" + str(inc),
                            input = self.phantom_labeler.inference_layers ["input-"+str(inc)].output,
                            input_shape = self.phantom_labeler.dropout_layers ["input-"+str(inc)].output_shape,
                            verbose = verbose )

            self.current.add_layer ( type = "unflatten",
                            id = "gan-input-unflattened-" + str(inc),
                            origin ="gan-input-" + str(inc),
                            shape = (28,28),
                            verbose = verbose
                            )

            self.current.add_layer ( type = "conv_pool",
                            id = "c1-gan-" + str(inc),
                            origin = "gan-input-unflattened-" + str(inc),
                            num_neurons = 20,
                            filter_size = (5,5),
                            pool_size = (2,2),
                            activation = 'relu',
                            regularize = True,  
                            batch_norm= True,       
                            input_params = self.current.dropout_layers ['c1'].params,                                                                    
                            verbose = verbose
                            )

            self.current.add_layer ( type = "conv_pool",
                            id = "c2-gan-" + str(inc),
                            origin = "c1-gan-" + str(inc),
                            num_neurons = 50,
                            filter_shape = (3,3),
                            pool_size = (2,2),
                            batch_norm= True,
                            regularize = True,                                                         
                            activation = 'relu',    
                            input_params = self.current.dropout_layers ['c2'].params,                                                                                                                
                            verbose = verbose
                            )

            self.current.add_layer ( type = "dot_product",
                            origin = "c2-gan-" + str(inc),
                            id = "fc1-gan-" + str(inc),
                            num_neurons = 800,
                            activation = 'relu',
                            batch_norm= True,
                            dropout_rate = 0.5,
                            regularize = True,                        
                            input_params = self.current.dropout_layers['fc1'].params,                        
                            verbose = verbose )

            self.current.add_layer ( type = "dot_product",
                            origin = "fc1-gan-" + str(inc),
                            id = "fc2-gan-" + str(inc),
                            num_neurons = 800,                    
                            activation = 'relu',
                            batch_norm= True,
                            dropout_rate = 0.5,
                            regularize = True,                        
                            input_params = self.current.dropout_layers['fc2'].params,                                                  
                            verbose = verbose  )                         
        
            self.current.add_layer ( type = "classifier",
                            id = "softmax-gan-" + str(inc),
                            origin = "fc2-gan-" + str(inc),
                            num_classes = self.num_classes,
                            activation = ('softmax', self.temperature),
                            input_params = self.current.dropout_layers \
                                             ['softmax'].params,      
                            regularize = True,                                                                       
                            verbose = verbose )  

            ##########
            # This will make the mentor values available to the current network so that we 
            # can caluclate errors
            ##########            
            self.current.add_layer(type = "random",
                                        id = "zero-targets-" + str(inc),
                                        num_neurons = (self.mini_batch_size, \
                                                        self.num_classes - old_w.shape[1] ), 
                                        distribution = 'binomial',
                                        p = 0,
                                        verbose = verbose)

            # input_shape = [self.phantom_labeler.layers['softmax-base-temperature'].output_shape,
            #            self.hallucinated.layers['zero-targets'].output_shape]                       

            # importing a layer from the mentor network. 
            self.current.add_layer (type = "tensor",
                                        id = "merge-import-" + str(inc),
                                        input = self.phantom_labeler.inference_layers \
                                                        ["phantom-" + str(inc)].output,
                                        input_shape = self.phantom_labeler.inference_layers \
                                                ["phantom-" + str(inc)].output_shape,
                                        verbose = verbose )

            self.current.add_layer (type = "merge",
                                        layer_type = "concatenate",
                                        id = "phantom-targets-" + str(inc),
                                        origin = ( "merge-import-" + str(inc),\
                                                    "zero-targets-" + str(inc)),
                                        verbose = verbose
                                        )
 


            ##########        
            # objective layers
            ##########

            # This is error between the temperature softmax layer and the mentor target.
            # This provides the incremental update.
            self.current.add_layer (type = "merge",
                            id = "obj-phantom-" + str(inc),
                            layer_type = "error",
                            error = "rmse",
                            origin = ("softmax-gan-" + str(inc), "phantom-targets-" + str(inc)),
                            )      
            objective_layers.append("obj-phantom-" +str(inc) )

        # This is the regular classifier objective for the incremental net.
        self.current.add_layer ( type = "objective",
                        id = "obj-current",
                        origin = "softmax",
                        verbose = verbose
                        )
        objective_layers.append("obj-current")

        # self.current.pretty_print()
        draw_network(self.current.graph, filename = 'hallucinated.png')    
        
        self.current.cook( optimizer = 'optim',
                objective_layers = objective_layers,
                datastream = 'data',
                classifier_layer = 'softmax',
                verbose = verbose
                )
    
    def update_base(self,
                    dataset = None,
                    root = '.',
                    params = None,
                    cook = False,
                    verbose = 1,
                    ):
        """
        This method updates the base network to keep up with current

        Args:
            Same as the setup base method.
        """
        if dataset is None:
            dataset = self.dataset[-1]
        
        ids = list()
        ids = self.base.get_params().keys()
        
        # Creating new copies of params again for the sake of 
        # seperation
        params = OrderedDict()
        current_params = self.current.get_params(verbose = verbose)                        

        for id in ids:
            params[id] = current_params[id]
        params = shared_params (params)
        
        self.base = self._mlp( dataset = dataset, 
                                params = params,
                                cook = False,
                                root = root,
                                id = 'base' + str(self.increment),
                                num_classes = self.num_classes,
                                verbose = verbose)
    


if __name__ == '__main__':
    pass