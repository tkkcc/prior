from util import *

def m():
    args = parse_args()
    args.n_stages = 10
    args.image='data/example1.png'
    args.kernel='data/example1.dlm'
    stage=args.n_stages
    b = ModelStack(stage)
    model=b
    m = b.m
    
    import h5py
    f = h5py.File('stages_01-10_finetuned.hdf5', 'r')
    for i in range(stage):
        a = f[f'model_{i+2}']
        for j in range(6):
            c = a[f'conv{j+1}_{i+1}']
            b= m[i].cnn[1+2*j]
            bias = torch.tensor(c['bias:0'].value)
            # may by 3,2,0,1
            weight = torch.tensor(c['kernel:0'].value).permute(3,2,1,0)
            b.bias.data.copy_(bias)
            b.weight.data.copy_(weight)
        for j in range(3):
            c = a[f'dense{j+1}_{i+1}']
            b = m[i].mlp[j*2]
            bias = torch.tensor(c['bias:0'].value)
            weight = torch.tensor(c['kernel:0'].value).t()
            b.bias.data.copy_(bias)
            b.weight.data.copy_(weight)
        c = a[f'lambda_{i+1}']
        b = m[i].mlp[6]
        bias = torch.tensor(c['bias:0'].value)
        weight = torch.tensor(c['kernel:0'].value).t()
        b.bias.data.copy_(bias)
        b.weight.data.copy_(weight)

        c = a[f'x_out_{i+1}/filter_weights:0']
        # may by t()
        weight = torch.tensor(c.value)
        m[i].fdn.filter_weights.data.copy_(weight)

    
    # parse arguments & setup
    # args = parse_args()
    if args.quiet:
        log = lambda *args,**kwargs: None
    else:
        def log(*args,**kwargs):
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

    if not args.quiet:
        log('Arguments:')
        pprint(vars(args))
    if args.output is None:
        import matplotlib.pyplot as plt
        if is_ipython():
            plt.ion()

    # load model config and do some sanity checks
    config = load_json(args.model_dir)
    n_stages = config['n_stages'] if args.n_stages is None else args.n_stages
    assert config['sigma_range'][0] <= args.sigma <= config['sigma_range'][1]
    assert 0 < n_stages <= config['n_stages']

    # load inputs
    img = img_as_float(imread(args.image)).astype(np.float32)
    if args.kernel.find('.') != -1 and os.path.splitext(args.kernel)[-1].startswith('.tif'):
        kernel = imread(args.kernel).astype(np.float32)
    else:
        kernel = np.loadtxt(args.kernel).astype(np.float32)
    if args.flip_kernel:
        kernel = kernel[::-1,::-1]
    kernel = np.clip(kernel,0,1)
    kernel /= np.sum(kernel)
    assert 2 <= img.ndim <= 3
    assert kernel.ndim == 2 and all([d%2==1 for d in kernel.shape])
    if img.ndim == 3:
        print('Warning: Applying grayscale deconvolution model to each channel of input image separately.',file=sys.stderr)

    # prepare for prediction
    log('Preparing inputs')
    t = pad_for_kernel(img, kernel, 'edge')
    t = edgetaper(t,kernel)
    y  = to_tensor(t)
    k  = np.tile(kernel[np.newaxis], (y.shape[0],1,1))
    s  = np.tile(args.sigma,(y.shape[0],1)).astype(np.float32)
    x0 = y

    # load models
    K.clear_session()
    log('Processing stages 01-%02d'%n_stages)
    log('- creating models and loading weights')
    weights = os.path.join(args.model_dir,'stages_01-%02d_%s.hdf5'%(n_stages,'finetuned' if args.finetuned else 'greedy'))
    if os.path.exists(weights):
        m = model_stacked(n_stages)
        m.load_weights(weights)
    else:
        assert not args.finetuned
        weights = [os.path.join(args.model_dir,'stage_%02d.hdf5'%(t+1)) for t in range(n_stages)]
        m = model_stacked(n_stages,weights)

    # predict
    log('tf predicting')
    pred = m.predict_on_batch([x0,y,k,s])
    result = crop_for_kernel(from_tensor(pred), kernel)
    del pred
    del m
    with torch.no_grad():
        log('torch predicting')
        [x0, y, k, s] = [torch.tensor(i, dtype=torch.float) for i in [x0, y, k, s]]
        result_ = model([x0, y, k, s]).detach().numpy()
        result_ = crop_for_kernel(from_tensor(result_),kernel)
        show(result_)
        print("\n==============", ((result - result_)**2).mean(), "============\n")
m()
