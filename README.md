# TransDNA



## Install


### Install for training 

- Clone the repo
``` sh
git clone https://github.com/qinyunnn/TansDNA.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n TransDNA python==3.8
conda activate TransDNA
pip install -r requirements.txt
```



## Usage

We provided a training example in examples.

## Data Preparation
```
# put your dataset under data/ directory with the same structure shown in the example/data/

data
  source_data
    |-reads.txt
    |-reference.txt
  target_data
    |-reads.txt
    |-reference.txt
```

### Notices:
* In reads.txt, the adjacent two clusters are segmented using '==============================='. For example:
```
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACACACGCAACCTCTTGCTGAAAGGCCT
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACAACGCAACCTCTTGCTGAAAGGCCT
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACACACGCAACCTCTTGCTGAAAGGCCT
AACCAATACCTTGAACCTAACTCGAGTTAACAAACGCAATTCACAGAACAAGGACGTCGGACGGTGTCCAGAATACCGGCCTCGTGACCGTGGCCAGGGAACCTGACAATGTCAGGCCTTACCGACACACGCAACCTCTTGCTGAAAGGCCT
===============================
AATACCTTGAAGTCACTGGTACTGAACATGCCTTCTGACCGTTAGGACTTATCTTCCTGTCGGTATAAGATCTACTACTACAACACTGGTTTCAACTAGCGGGAGAAGTCCTTACCGAGTTCTGCGGCTGGCTGATAGCGTGTGCCCTCTGG
AATACCTTGAAGTCACTGGTACTGAACATGCCTTCTGACCGTTAGGACTTATCTTCCTGTCGGTATAAGATCTACTACTACAACACTGGTTTCAACTAGCGGGAGAAGTCCTTACCGAGTTCTGCGGCTGGCTGATAGCGTGTGCCCTCTGG
AATACCTTGAAGTCACTGGTACTGAACATGCCTTCTGACCGTTAGGACTTATCTTCCTGTCGGTATAAGATCTACTACTACAACACTGGTTTCAACTAGCGGGAGAAGTCCTTACCGAGTTCTGCGGCTGGCTGATAGCGTGTGCCCTCTGG
===============================
```


## Run
* First, you can train the source domain with run_source.sh to obtain the pre-trained encoder and decoder.
```
    cd examples
    bash run_source.sh
```
* Then, transfer learning is performed using run_target.sh.
```
    bash run_target.sh
```
You can change running parameters in run_source.sh and run_target.sh.
