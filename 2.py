from util import *

def m():
    args = parse_args()
    log = lambda *args, **kwargs: None
    n_stages = args.n_stages
    size=100
    # img = img_as_float(imread(args.image)).astype(np.float32)
    img = img_as_float(imread("data/example2.png")).astype(np.float32)
    img = resize(img, (size, size)).astype(np.float32)
    # gt = img_as_float(imread("data/example2_.png")).astype(np.float32)
    # gt = resize(gt, (size, size)).astype(np.float32)
    # gt=img + np.random.normal(0, 25/255.0, img.shape).astype(np.float32)
    # gt,img = img,gt
    gt = img+5

    kernel = np.loadtxt('data/example2.dlm').astype(np.float32)
    kernel = np.clip(kernel, 0, 1)
    kernel /= np.sum(kernel)

    # y = to_tensor(edgetaper(pad_for_kernel(img, kernel, "edge"), kernel))
    # k = np.tile(kernel[np.newaxis], (y.shape[0], 1, 1))
    # s = np.tile(args.sigma, (y.shape[0], 1)).astype(np.float32)
    # x0 = y
    # img = np.random.rand(10, 12, 3).astype(np.float32)

    # y = to_tensor(edgetaper(pad_for_kernel(img, kernel, "edge"), kernel))
    y = to_tensor(img)
    k = np.tile(kernel[np.newaxis], (y.shape[0], 1, 1))
    s = np.tile(args.sigma, (y.shape[0], 1)).astype(np.float32)
    x0 = y
    gt=to_tensor(gt)
    # kernel = np.random.rand(3, 3).astype(np.float32)
    # y = to_tensor(img)
    # k = np.tile(kernel[np.newaxis], (y.shape[0], 1, 1))
    # s = np.tile(1.5, (y.shape[0], 1)).astype(np.float32)
    # x0 = y

    epoch = 20
    stage = 5
    lr = 0.3
    # =============tf
    # K.clear_session()
    # a = model_stacked(stage)
    # # a = model_stage(1)
    # # a = model_stage(1).predict_on_batch([x0, y, k, s]).astype(np.float32)
    # sgd = keras.optimizers.SGD(lr=lr)
    # a.compile(loss="mean_squared_error", optimizer=sgd)

    # a.fit([x0, y, k, s], gt, epochs=epoch, batch_size=3)
    # a = a.predict_on_batch([x0, y, k, s]).astype(np.float32)
    # print(a.shape)

    b = ModelStack(stage)
    [x0, y, k, s, gt] = [torch.tensor(i, dtype=torch.float) for i in [x0, y, k, s, gt]]
    if torch.cuda.is_available():
        [x0, y, k, s,gt,b] = [i.cuda() for i in [x0, y, k, s,gt,b]]
    # b = FDN()([x0, x0, y, k, s]).numpy().astype(np.float32)
    # b = M()([x0, y, k, s]).detach().numpy().astype(np.float32)
    # b = M()

    optimizer = torch.optim.SGD(b.parameters(), lr=lr)
    mse = nn.MSELoss()
    # a=mse(gt, torch.zeros_like(gt))
    # b=gt.pow(2).sum()
    # print(b.m[0].fdn.filter_weights.data.sum())
    for i in range(epoch):
        optimizer.zero_grad()
        out = b([x0, y, k, s])
        loss = mse(out, gt)
        print(loss)
        loss.backward()
        optimizer.step()
    # print(b.m[0].fdn.filter_weights.data.sum())
    with torch.no_grad():
        b = b([x0, y, k, s]).detach().numpy()
    print(b.shape)
    # show(b)
    # print("\n==============", compare_psnr(a,b), "============\n")
    if 'a' in vars():
        print("\n==============", ((a - b)**2).mean(), "============\n")


m()
