from yann.special.datasets import combine_split_datasets_train_only as combine
from dataset import cook_cifar10_complete, cook_svhn_complete

dataset1 = cook_cifar10_complete()
dataset2 = cook_svhn_complete()
loc = ( dataset1.dataset_location(), dataset2.dataset_location() )

# loc = ('_datasets/_dataset_65095', '_datasets/_dataset_866')
combine (loc)