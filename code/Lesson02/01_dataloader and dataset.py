import torch

'''
Data:

1. Data gathering: Img, Label
2. Data division: train valid, test
3. Data loading: DataLoader:
    a. Sampler: index
    b. DataSet: Img, Label
4. Data preprocessing: transforms
    
'''

'''
1. torch.utils.data.DataLoader

    torch.utils.data.DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                sampler=None,
                                batch_sampler=None,
                                num_workers=0,
                                collate_fn=None,
                                pin_memory=false,
                                drop_last=False,
                                timeout=0,
                                worker_init_fn=None,
                                multiprocessing_context=None)
                                
    Function: construct iterable data loader
    
    Parameters:
        a. dataset: Dataset class, determines where to read and how to read the data
        b. batchsize: the size of each batch
        c. num_work: whether reading data in multi thread
        d. shuffle: whether shuffle each epoch
        e. drop_last: when the number of sample is not divisible, whether dropping the last batch of data
        
'''
