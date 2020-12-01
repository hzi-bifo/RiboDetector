## `RiboDetector` - Accurate and rapid RiboRNA sequences Detector based on deep learning

### About `Ribodetector`

`RiboDetector` is a tool developed to accurately yet rapidly detect and remove rRNA sequences from metagenomeic, metatranscriptomic, and ncRNA sequencing data. It was developed based on LSTM and optimized for both GPU and CPU to achieve a **10** times on CPU and **50** times on a consumer GPU faster runtime compared to the current state-of-the-art tools. Moreover, it is way more accurate than any other tools, by having ~**10** times less false classifications. Besides, the size of RiboDetector is only few MBs which is over ~**1000** times smaller than commanly used tools.

### Prerequirements

To be able to use `RiboDetector`, all you need to do is to install `Python 3.8` (make sure you have version `3.8` because `3.7` cannot serialize a string larger than 4GiB) with `conda` and Python libraries: `tqdm`, `numpy`, `biopython`, `torch` or `onnxruntime` (if run on CPU mode) using `pip`, for example:

```shell
conda create -n ribodetector python=3.8
conda activate ribodetector
pip install tqdm numpy biopython torch torchvision onnxruntime
```
**Note**: To instal torch compatible with your CUDA version, please fellow this instruction:
https://pytorch.org/get-started/locally/

### Usage

#### GPU mode

```shell
usage: detect.py [-h] [-c CONFIG] [-d DEVICEID] -l LEN -i [INPUT [INPUT ...]] -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]]
                 [-e {rrna,norrna,both}] [-t THREADS] [-m MEMORY] [--chunk_size CHUNK_SIZE]

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
  -e {rrna,norrna,both}, --ensure {rrna,norrna,both}
                        Only output certain sequences with high confidence
                        norrna: output non-rRNAs with high confidence, remove as many rRNAs as possible;
                        rrna: vice versa, output rRNAs with high confidence;
                        both: both non-rRNA and rRNA prediction with high confidence.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 10)
  -m MEMORY, --memory MEMORY
                        amout (GB) of GPU RAM. (default: 12)
  --chunk_size CHUNK_SIZE
                        Use this parameter when having low memory. Parsing the file in chunks.
                        Not needed when free RAM >=5 * your_file_size (uncompressed, sum of paired ends).
                        When chunk_size=256, memory=16 it will load 256 * 16 * 1024 reads each chunk (use ~20 GB for 100bp paired end).
```

#### CPU mode

```shell
usage: detect_cpu.py [-h] [-c CONFIG] -l LEN -i [INPUT [INPUT ...]] -o [OUTPUT [OUTPUT ...]] [-r [RRNA [RRNA ...]]] [-e {rrna,norrna,both}] [-t THREADS] [--chunk_size CHUNK_SIZE]

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
  -e {rrna,norrna,both}, --ensure {rrna,norrna,both}
                        Only output certain sequences with high confidence
                        norrna: output non-rRNAs with high confidence, remove as many rRNAs as possible;
                        rrna: vice versa, output rRNAs with high confidence;
                        both: both non-rRNA and rRNA prediction with high confidence.
                              (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)
  -t THREADS, --threads THREADS
                        number of threads to use. (default: 10)
  --chunk_size CHUNK_SIZE
                        chunk_size * threads reads to process per thread.
                        When chunk_size=1024 and threads=10, each process will load 1024 reads, in total 1024 * 10 reads consumming 10G memory.
```

### Benchmarks
