from util import dotdict

o = {
    "model": "FDN",
    "device": "cuda",
    "batch_size": 3,
    "num_workers": 0,
    "epoch": 10,
    "lr": 0.001,
}

o=dotdict(o)

if __name__ == '__main__':
    o.update({'device':'cpu'})
# {
#     env = 'default' # visdom 环境
#     model = 'FDN' # 使用的模型，名字必须与models/__init__.py中的名字一致
#     device ='cuda'
#     train_data_root = './data/train/' # 训练集存放路径
#     test_data_root = './data/test1' # 测试集存放路径
#     load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

#     batch_size = 128 # batch size
#     use_gpu = True # user GPU or not
#     num_workers = 4 # how many workers for loading data
#     print_freq = 20 # print info every N batch

#     debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
#     result_file = 'result.csv'

#     max_epoch = 10
#     lr = 0.1 # initial learning rate
#     lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
#     weight_decay = 1e-4  #
# }
