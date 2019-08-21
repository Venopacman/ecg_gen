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

# Research plan 
- [x] Find data
- [ ] Find architecture capable for generation ecg-like structures
- [ ] Develop GAN for short 1-lead ecg generation
- [ ] Develop GAN for medium 1-lead ecg generation
- [ ] Develop GAN for long 1-lead ecg generation
- [ ] Develop GAN fro short n-lead ecg generation
- [ ] Plan further work