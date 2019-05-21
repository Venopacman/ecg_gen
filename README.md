# ecg_gen
ECG generation with Conditional GANs


# Installation
```
apt install virtualenv
git clone https://github.com/Venopacman/ecg_gen.git
cd esn_gen
virtualenv --python=python3.6 ./venv
source venv/bin/activate
make install

git clone https://github.com/josipd/torch-two-sample.git
cd torch-two-sample
python setup.py install

```

# Gan training
```
make train_gan 
```