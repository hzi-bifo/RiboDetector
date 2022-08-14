## RiboDetector - Accurate and rapid RiboRNA sequences Detector based on deep learning

### About Ribodetector
<img src="RiboDetector_logo.png" width="600" />

`RiboDetector` is a software developed to accurately yet rapidly detect and remove rRNA sequences from metagenomeic, metatranscriptomic, and ncRNA sequencing data. It was developed based on LSTMs and optimized for both GPU and CPU usage to achieve a **10** times on CPU and **50** times on a consumer GPU faster runtime compared to the current state-of-the-art software. Moreover, it is very accurate, with ~**10** times fewer false classifications. Finally, it has a low level of bias towards any GO functional groups. 


### Prerequirements

#### 1. Create `conda` env and install `Python v3.8`

To be able to use `RiboDetector`, you need to install `Python v3.8` or `v3.9` (make sure you have version `3.8` because `3.7` cannot serialize a string larger than 4GiB) with `conda`:
```shell
conda create -n ribodetector python=3.8
conda activate ribodetector
```

#### 2. Install `pytorch` in the ribodetector env if GPU is available

To install `pytorch` compatible with your CUDA version, please fellow this instruction:
https://pytorch.org/get-started/locally/. Our code was tested with `pytorch v1.7`, `v1.7.1`, `v1.10.2`.

Note: you can skip this step if you don't use GPU

### Installation

#### Using pip

```shell
pip install ribodetector
```

#### Using conda
```shell
conda install -c bioconda ribodetector
```

### Usage

#### GPU mode

#### Example
```shell
ribodetector -t 20 \
  -l 100 \
  -i inputs/reads.1.fq.gz inputs/reads.2.fq.gz \
  -m 10 \
  -e rrna \
  --chunk_size 256 \
  -o outputs/reads.nonrrna.1.fq outputs/reads.nonrrna.2.fq
```
The command lind above excutes ribodetector for paired-end reads with length 100 using GPU and 20 CPU cores

#### Full help
```shell
usage: ribodetector [-h] [-c CONFIG] [-d DEVICEID] -l LEN -i [INPUT [INPUT ...]]
  -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]] [-e {rrna,norrna,both,none}] 
  [-t THREADS] [-m MEMORY] [--chunk_size CHUNK_SIZE] [-v]

rRNA sequence detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path of config file
  -d DEVICEID, --deviceid DEVICEID
                        Indices of GPUs to enable. Quotated comma-separated device ID numbers. (default: all)
  -l LEN, --len LEN     Sequencing read length. Note: the accuracy reduces for reads shorter than 40.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Path of input sequence files (fasta and fastq), the second file will be considered 
                        as second end if two files given.
  -o [OUTPUT [OUTPUT ...]], --output [OUTPUT [OUTPUT ...]]
                        Path of the output sequence files after rRNAs removal (same number of files as input).
                        (Note: 2 times slower to write gz files)
  -r [RRNA [RRNA ...]], --rrna [RRNA [RRNA ...]]
                        Path of the output sequence file of detected rRNAs (same number of files as input)
  -e {rrna,norrna,both,none}, --ensure {rrna,norrna,both,none}
                        Ensure which classificaion has high confidence for paired end reads.
                        norrna: output only high confident non-rRNAs, the rest are clasified as rRNAs;
                        rrna: vice versa, only high confident rRNAs are classified as rRNA and the rest output as non-rRNAs;
                        both: both non-rRNA and rRNA prediction with high confidence;
                        none: give label based on the mean probability of read pair.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 10)
  -m MEMORY, --memory MEMORY
                        amount (GB) of GPU RAM. (default: 12)
  --chunk_size CHUNK_SIZE
                        Use this parameter when having low memory. Parsing the file in chunks.
                        Not needed when free RAM >=5 * your_file_size (uncompressed, sum of paired ends).
                        When chunk_size=256, memory=16 it will load 256 * 16 * 1024 reads each chunk (use ~20 GB for 100bp paired end).
  -v, --version         show program's version number and exit
```

#### CPU mode

#### Example
```shell
ribodetector_cpu -t 20 \
  -l 100 \
  -i inputs/reads.1.fq.gz inputs/reads.2.fq.gz \
  -e rrna \
  --chunk_size 256 \
  -o outputs/reads.nonrrna.1.fq outputs/reads.nonrrna.2.fq
```
The above command line excutes ribodetector for paired-end reads with length 100 using 20 CPU cores.

When using SLURM job submission system, you need to specify `--cpus-per-task` to the number you CPU cores you need and set `--threads-per-core` to 1.

#### Full help

```shell

usage: ribodetector_cpu [-h] [-c CONFIG] -l LEN -i [INPUT [INPUT ...]] 
  -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]] [-e {rrna,norrna,both,none}] 
  [-t THREADS] [--chunk_size CHUNK_SIZE] [-v]

rRNA sequence detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path of config file
  -l LEN, --len LEN     Sequencing read length. Note: the accuracy reduces for reads shorter than 40.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Path of input sequence files (fasta and fastq), the second file will be considered as 
                        second end if two files given.
  -o [OUTPUT [OUTPUT ...]], --output [OUTPUT [OUTPUT ...]]
                        Path of the output sequence files after rRNAs removal (same number of files as input).
                        (Note: 2 times slower to write gz files)
  -r [RRNA [RRNA ...]], --rrna [RRNA [RRNA ...]]
                        Path of the output sequence file of detected rRNAs (same number of files as input)
  -e {rrna,norrna,both,none}, --ensure {rrna,norrna,both,none}
                        Ensure which classificaion has high confidence for paired end reads.
                        norrna: output only high confident non-rRNAs, the rest are clasified as rRNAs;
                        rrna: vice versa, only high confident rRNAs are classified as rRNA and the rest output as non-rRNAs;
                        both: both non-rRNA and rRNA prediction with high confidence;
                        none: give label based on the mean probability of read pair.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 20)
  --chunk_size CHUNK_SIZE
                        chunk_size * 1024 reads to load each time.
                        When chunk_size=1000 and threads=20, consumming ~20G memory, better to be multiples of the number of threads..
  -v, --version         show program's version number and exit
```

**Note**: RiboDetector uses multiprocessing with shared memory, thus the memory use of a single process indicated in `htop` or `top` is actually the total memory used by RiboDector. Some job submission system like SGE mis-calculated the total memory use by adding up the memory use of all process. If you see this do not worry it will cause out of memory issue. 

<!-- ### Benchmarks

We benchmarked five different rRNA detection methods including RiboDetector on 8 benchmarking datasets as following: 

- 20M paired end reads simulated based on  rRNA sequences from Silva database, those sequences are distinct from sequences used for training and validation.

- 20M paired end reads simulated based on 500K CDS sequences from OMA databases.

- 27,206,792 paired end reads simulated based on 13,848 viral gene sequences downloaded from ENA database.

- 7,917,920 real paired end amplicon sequencing reads targeting V1-V2 region  of  16s rRNA genes from oral microbiome study.

- 6,330,381 paired end reads simulated from 106,880 human noncoding RNA sequences.

- OMA_Silva dataset in figure C contains 1,027,675 paired end reads simulated on CDS sequences which share similarity to rRNA genes, the sequences with identity >=98% and query coverage >=90% to rRNAs were excluded.

- HOMD dataset in figure C has 100,558 paired end reads simulated on CDS sequences from HOMD database which share similarity to the FP sequences of three tools, again sequences with identity >=98% and query coverage >=90% to rRNAs were excluded.

- GO_FP_N_02 in figure C consisting of 678,250 paired end reads was simulated from OMA sequences which have the GO with FP reads ratio >=0.2 on 20M mRNA reads dataset for BWA, RiboDetector or SortMeRNA.

![Benchmarking the performance and runtime of different rRNA sequences detection methods](./benchmarks/benchmarks.jpg)

In the above figures, the definitions of *FPNR* and *FNR* are:

<img src="https://render.githubusercontent.com/render/math?math=\large FPNR=100\frac{false \:predictions}{total \: sequences}">

<img src="https://render.githubusercontent.com/render/math?math=\large FNR=100\frac{false \:negatives}{total \:positives}">

RiboDetector has a very high generalization ability and is capable of detecting novel rRNA sequences (Fig. C). -->


### Citation
Deng ZL, Münch PC, Mreches R, McHardy AC. Rapid and accurate detection of ribosomal RNA sequences using deep learning. <i>Nucleic Acids Research</i>. 2022. (https://doi.org/10.1093/nar/gkac112)

### Acknowledgements
The scripts from the `base` dir were from the template [pytorch-template
](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque) and other [contributors](https://github.com/victoresque/pytorch-template/graphs/contributors).
