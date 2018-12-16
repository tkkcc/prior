## dependency

python3.4+ torch numpy skimage scipy matplotlib tqdm

## prepare

```sh
# download BSD500
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/fdn1201_data.zip
unzip -n fdn1201_data.zip
# download Levin
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/LevinEtalCVPR09Data.rar
unrar x LevinEtalCVPR09Data.rar data/
# download Sun
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/input80imgs8kernels.zip
unzip -n input80imgs8kernels.zip -d data/
```

## structure

data:

- BSR: [BSD500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
- LevinEtalCVPR09Data: [Levin](http://webee.technion.ac.il/people/anat.levin/papers/LevinEtalCVPR09Data.rar)
- kernel: 8 blur kernels from Sun's [deblur2013iccp](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html), [deblur_iccp2013_testset_640](http://cs.brown.edu/~lbsun/deblur2013/deblur_iccp2013_testset_640.zip)
- input80imgs8kernels: [Sun](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html)

model:

- fdn: pytorch port of [fdn](https://github.com/uschmidt83/fourier-deconvolution-network)
- fdn2: our model using closed form method
- fdn3: our model using sgd method

save:

- 01f.tar: finetuned 01 stage from tf port
- 01-10g.tar: trained using BSD3000, just one stage

entry:

- 2.py, reproduce FDN, BSD3000

## extra

[fourier-deconvolution-network](https://github.com/uschmidt83/fourier-deconvolution-network)

[Understanding and evaluating blind deconvolution algorithms](http://webee.technion.ac.il/people/anat.levin/)

[Edge-based blur kernelestimation using patch priors](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html)


