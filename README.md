# SuperAdvNet
Implementation for paper 《Interpreting the Correlation between Adversarial Robustness  and  Compression Variance of Deep Learning Models in IoT Domain》

## Intoruction
The originization of this code is shown as follow:
```
|-- README.md # short introduction of codes
|-- figures # the original figure in the Paper
|   |-- cifar10
    `-- cifar100
|-- prototype
|   -- prototype  # the python implementation of this project, including NAS supernet for compression and adversirial testing
        |-- __init__.py
        |-- data
        |-- loss_functions
        |-- lr_scheduler
        |-- model
        |-- optimizer
        |-- solver
        |-- spring
        `-- utils
|-- requirements.txt # the env needs of our codes
`-- workspace
    |-- bignas_adv_cifar10
    `-- bignas_adv_cifar100
```

## Usage
1. First, install the required enviromental setup.
```
pip install -r requirements.txt
```

2. Go to the workspace dir of cifar10 and cifar100, and enjoy it:)
```
cd bignas_adv_cifar10         # or (bignas_adv_cifar100)
bash train.sh                 # train the superAdvNet, save the weight of superAdvNet
cd Adv_Evaluate               # or (Adv_Sample_Accuracy)
bash run.sh                   # get the adv result generated from the same model
cd ../Max_Adv_Evaluate        # or (Max_Adv__Sample_Accuracy)
bash run.sh                   # get the adv result generated from uncompressed model
cd ../Min_Adv_Evaluate        # or (Min_Adv_Sample_Accuracy)
bash run.sh                   # get the adv result generated from fully compressed model
```

