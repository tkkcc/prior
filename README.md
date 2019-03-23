## dependency
python3.4+ torch numpy scikit-image scipy matplotlib tqdm tensorboardX

## dataset
```sh
# BSD500
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
tar xvfz BSR_bsds500.tgz
mv BSR/BSDS500/data/images data/BSD500
# Levin
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/LevinEtalCVPR09Data.rar
unrar x LevinEtalCVPR09Data.rar data/
# Sun
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/input80imgs8kernels.zip
unzip -n input80imgs8kernels.zip -d data/
# TNRD 400
wget https://www.dropbox.com/s/8j6b880m6ddxtee/TNRD-Codes.zip
unzip TNRD-Codes.zip
mv TNRD-Codes/TrainingCodes4denoising/FoETrainingSets180 data/
# BSD68
# mv "TNRD-Codes/TestCodes(denoising-deblocking-SR)/GaussianDenoising/68imgs" data/BSD68
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/BSD68.tar
tar xvf BSD68.tar -C data/
# Waterloo Exploration 4744
wget http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar
unrar x exploration_database_and_code.rar
mv exploration_database_and_code/pristine_images data/WED4744
# Set 12
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/Set12.zip
unzip -n Set12.zip -d data/
# ILSVRC12 random 400
wget https://github.com/tkkcc/tmp/releases/download/0.0.1/ILSVRC12.tar
tar xvf ILSVRC12.tar -C data/
```

## source
- 0.sh: main script
- 2.py: main entry
- 1.py: visualize and evaluate function
- 1.sh: random copy images from ILSVRC12/train to data/ILSVRC12

## run
```sh
./0.sh teston Set12
```