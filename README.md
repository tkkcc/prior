## dependency

python3.4+ torch numpy skimage scipy matplotlib tqdm

## prepare

```sh
# download BSD500
wget "https://github.com/tkkcc/tmp/releases/download/0.0.1/fdn1201_data.zip"
# extract to data/ without override
unzip -n fdn1201_data
```

## structure

data:

- BSR: [BSD500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- kernel: 8 blur kernels from Sun's [deblur2013iccp](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html), [deblur_iccp2013_testset_640](http://cs.brown.edu/~lbsun/deblur2013/deblur_iccp2013_testset_640.zip)

model:

- fdn: pytorch port of https://github.com/uschmidt83/fourier-deconvolution-network
- fdn2: our model using closed form method
- fdn3: our model using sgd method

entry:

```sh
# train using BSD, mse loss, Adam
python 1.py
```

## extra

[fourier-deconvolution-network](https://github.com/uschmidt83/fourier-deconvolution-network)

Understanding and evaluating blind deconvolution algorithms

[http://webee.technion.ac.il/people/anat.levin/](http://webee.technion.ac.il/people/anat.levin/)

[http://webee.technion.ac.il/people/anat.levin/papers/LevinEtalCVPR09Data.rar](http://webee.technion.ac.il/people/anat.levin/papers/LevinEtalCVPR09Data.rar)

Edge-based blur kernelestimation using patch priors

[http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html)

[http://cs.brown.edu/~lbsun/deblur2013/deblur_iccp2013_testset_640.zip](http://cs.brown.edu/~lbsun/deblur2013/deblur_iccp2013_testset_640.zip)
