import os
import random
import shutil

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
        
    
    Epoch: one forward pass and one backward pass of all the training examples
    Batch Size: the number of training examples in one forward/backward pass.
                The higher the batch size, the more memory space you'll need.
    Number of iterations = number of passes, each pass using [batch size] number of examples.
                To be clear, one pass = one forward pass + one backward pass.
                We do not count the forward pass and backward pass as two different passes.
    Examples:
        a. Sample: 80; Batchsize: 8; 1 Epoch = 10 Iteration
        b. Sample: 87, Batchsize: 8; 
            (i) drop_last = True: 1 Epoch = 10 Iteration
            (ii) drop_last = False: 1 Epoch = 11 Iteration
    
2. torch.utils.data.Dataset

    class Dataset(object):
        
        def __getitem__(self, index):
            raise NotImplementedError
            
        def __add__(self, order):
            return ConcatDataset([self, other])
            
    Function: Dataset abstract class, every self-definition Dataset should inherit it and rewrite.
    
    __getitem__()
    getitem: receive an index and return a sample
'''


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)

    dataset_dir = os.path.join("..", "..", "data", "RMB_data")
    split_dir = os.path.join("..", "..", "data", "RMB_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files, in os.walk(dataset_dir):
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point,
                                            valid_point-train_point, img_count-valid_point))





