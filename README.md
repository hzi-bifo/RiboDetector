## `RiboDetector` - Accurate and rapid RiboRNA sequences Detector based on deep learning

### About `Ribodetector`

`RiboDetector` is a software developed to accurately yet rapidly detect and remove rRNA sequences from metagenomeic, metatranscriptomic, and ncRNA sequencing data. It was developed based on LSTMs and optimized for both GPU and CPU usage to achieve a **10** times on CPU and **50** times on a consumer GPU faster runtime compared to the current state-of-the-art software. Moreover, it is very accurate, with ~**10** times fewer false classifications. Finally, it has a low level of bias towards any GO functional groups. 


### Prerequirements

To be able to use `RiboDetector`, all you need to do is to install `Python 3.8` (make sure you have version `3.8` because `3.7` cannot serialize a string larger than 4GiB) with `conda` and Python libraries: `tqdm`, `numpy`, `biopython`, `pandas`, `torch` or `onnxruntime` (if run on CPU mode) using `pip`, for example:

```shell
conda create -n ribodetector python=3.8
conda activate ribodetector
pip install tqdm numpy pandas biopython torch torchvision onnxruntime
```
**Note**: To install torch compatible with your CUDA version, please fellow this instruction:
https://pytorch.org/get-started/locally/

### Usage

#### GPU mode

```shell
usage: detect.py [-h] [-c CONFIG] [-d DEVICEID] -l LEN -i [INPUT [INPUT ...]] -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]] [-e {rrna,norrna,both,none}] [-t THREADS] [-m MEMORY]
                 [--chunk_size CHUNK_SIZE] [-v]

rRNA sequence detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path of config file
  -d DEVICEID, --deviceid DEVICEID
                        Indices of GPUs to enable. Quotated comma-separated device ID numbers. (default: all)
  -l LEN, --len LEN     Sequencing read length, should be not smaller than 50.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Path of input sequence files (fasta and fastq), the second file will be considered as second end if two files given.
  -o [OUTPUT [OUTPUT ...]], --output [OUTPUT [OUTPUT ...]]
                        Path of the output sequence files after rRNAs removal (same number of files as input).
                        (Note: 2 times slower to write gz files)
  -r [RRNA [RRNA ...]], --rrna [RRNA [RRNA ...]]
                        Path of the output sequence file of detected rRNAs (same number of files as input)
  -e {rrna,norrna,both,none}, --ensure {rrna,norrna,both,none}
                        Only output certain sequences with high confidence
                        norrna: output non-rRNAs with high confidence, remove as many rRNAs as possible;
                        rrna: vice versa, output rRNAs with high confidence;
                        both: both non-rRNA and rRNA prediction with high confidence;
                        none: give label based on the mean probability of read pair.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 10)
  -m MEMORY, --memory MEMORY
                        amout (GB) of GPU RAM. (default: 12)
  --chunk_size CHUNK_SIZE
                        Use this parameter when having low memory. Parsing the file in chunks.
                        Not needed when free RAM >=5 * your_file_size (uncompressed, sum of paired ends).
                        When chunk_size=256, memory=16 it will load 256 * 16 * 1024 reads each chunk (use ~20 GB for 100bp paired end).
  -v, --version         show program's version number and exit
```

#### CPU mode

```shell
usage: detect_cpu.py [-h] [-c CONFIG] -l LEN -i [INPUT [INPUT ...]] -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]] [-e {rrna,norrna,both,none}] [-t THREADS] [--chunk_size CHUNK_SIZE] [-v]

rRNA sequence detector

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path of config file
  -l LEN, --len LEN     Sequencing read length, should be not smaller than 50.
  -i [INPUT [INPUT ...]], --input [INPUT [INPUT ...]]
                        Path of input sequence files (fasta and fastq), the second file will be considered as second end if two files given.
  -o [OUTPUT [OUTPUT ...]], --output [OUTPUT [OUTPUT ...]]
                        Path of the output sequence files after rRNAs removal (same number of files as input).
                        (Note: 2 times slower to write gz files)
  -r [RRNA [RRNA ...]], --rrna [RRNA [RRNA ...]]
                        Path of the output sequence file of detected rRNAs (same number of files as input)
  -e {rrna,norrna,both,none}, --ensure {rrna,norrna,both,none}
                        Only output certain sequences with high confidence
                        norrna: output non-rRNAs with high confidence, remove as many rRNAs as possible;
                        rrna: vice versa, output rRNAs with high confidence;
                        both: both non-rRNA and rRNA prediction with high confidence;
                        none: give label based on the mean probability of read pair.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 10)
  --chunk_size CHUNK_SIZE
                        chunk_size * threads reads to process per thread.(default: 256)
                        When chunk_size=1024 and threads=10, each process will load 1024 reads, in total consumming 10G memory.
  -v, --version         show program's version number and exit
```

### Benchmarks

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

RiboDetector has a very high generalization ability and is capable of detecting novel rRNA sequences (Fig. C).

### Acknowledgements
The scripts from the `base` dir were from the template [pytorch-template
](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque) and other [contributors](https://github.com/victoresque/pytorch-template/graphs/contributors).
