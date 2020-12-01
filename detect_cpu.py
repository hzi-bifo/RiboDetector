import os
import math
import gzip
import argparse
import itertools
import onnxruntime
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
from collections import defaultdict
from parse_config import ConfigParser

from argparse import RawTextHelpFormatter
import data_loader.seq_encoder as SeqEncoder

# Get the directory of the program
cd = os.path.dirname(os.path.abspath(__file__))


def open_for_write(read_file):
    """Open a plain text or gzipped text file for writing according to the file name extension

    Args:
        read_file (str): the file name to write

    Returns:
        file handle: the opened file handle for writing
    """
    if read_file.endswith('gz'):
        return gzip.open(read_file, mode='wt', compresslevel=5)
    else:
        return open(read_file, 'w')


class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    OKYELLOW = '\033[33m'
    OKMAG = '\033[35m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    UPDATE = '\033[F'


class Predictor:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.logger = config.get_logger('predict', 1)
        # self.batch_size = self.args.memory * 1024
        self.chunk_size = self.args.chunk_size
        self.input = self.args.input
        self.output = self.args.output
        self.rrna = self.args.rrna
        self.SENTINEL = 1  # progressbar signal

    def load_model(self):
        if self.args.len < 50:
            self.logger.error('{}Sequence length is too short to classify!{}'.format(
                colors.FAIL,
                colors.ENDC))
            raise RuntimeError(
                "Sequence length must be set to larger than 50.")
        elif 50 <= self.args.len < 100:
            self.len = 50
        elif 100 <= self.args.len < 150:
            self.len = 100
        else:
            self.len = 150
        if self.args.ensure == 'norrna':
            model_file_ext = 'recall'
        else:
            model_file_ext = 'mcc'

        self.model_file = os.path.join(
            cd, self.config['state_file']['read_len{}_{}'.format(self.len, model_file_ext)]).replace('.plt', '.onnx')

        self.model = onnxruntime.InferenceSession(self.model_file)

        self.logger.info('Using high {} model file: {}{}{}{} on CPU'.format(model_file_ext.upper(),
                                                                            colors.BOLD,
                                                                            colors.OKCYAN,
                                                                            self.model_file,
                                                                            colors.ENDC))

    def run(self):
        num_workers = self.args.threads
        num_inputs = len(self.input)
        num_rrna_outputs = None if self.rrna is None else len(self.rrna)
        num_norrna_outputs = len(self.output)
        if num_inputs != num_norrna_outputs or num_inputs > 2:
            self.logger.error('{}The number of input and output sequence files is invalid!{}'.format(
                colors.FAIL,
                colors.ENDC))
            raise RuntimeError(
                "Input or output should have no more than two files and they should have the same number of files.")
        if num_rrna_outputs != None and num_rrna_outputs != num_inputs:
            self.logger.error('{}The number of output rRNA sequence files is invalid!{}'.format(
                colors.FAIL,
                colors.ENDC))
            raise RuntimeError(
                "Ouput rRNA should have no more than two files and they should the same number with input files.")

        is_paired = (num_inputs == 2)

        manager = mp.Manager()
        results = manager.list()
        work = manager.Queue(num_workers)

        pool = []

        # queue for progressbar signal
        q_pbar = mp.Queue()

        if is_paired:
            with mp.Pool(2) as p:
                input_reads = p.map(SeqEncoder.load_reads, self.input)

            num_seqs = len(input_reads[0])

            self.num_chunks = math.ceil(
                num_seqs / (self.chunk_size))

            self.logger.info('{}{}{}{} sequences loaded!'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_seqs,
                colors.ENDC))

            if self.rrna is not None:
                self.logger.info('Writing output rRNA sequences into file: {}{}{}'.format(
                    colors.OKBLUE,
                    ", ".join(self.rrna),
                    colors.ENDC))
                rrna1_fh = open_for_write(self.rrna[0])
                rrna2_fh = open_for_write(self.rrna[1])

            self.logger.info('Writing output non-rRNA sequences into file: {}{}{}'.format(
                colors.OKBLUE,
                ", ".join(self.output),
                colors.ENDC))

            norrna1_fh = open_for_write(self.output[0])
            norrna2_fh = open_for_write(self.output[1])

            if self.args.ensure == 'both':

                unclf1 = self.output[0] + '.unclassified.gz'
                unclf2 = self.output[1] + '.unclassified.gz'
                unclf1_fh = open_for_write(unclf1)
                unclf2_fh = open_for_write(unclf2)
                self.logger.info('Writing unclassified sequences into file: {}{}, {}{}'.format(
                    colors.OKYELLOW,
                    unclf1,
                    unclf2,
                    colors.ENDC))

            num_rrna = 0
            num_unknown = 0

            proc = mp.Process(target=self.listener, args=(q_pbar,))
            proc.start()

            for i in range(num_workers):
                p = mp.Process(target=self.classify_paired_reads,
                               args=(work, results, q_pbar))
                p.start()
                pool.append(p)

            iters = itertools.chain(Predictor.generate_paired_read_chunks(
                input_reads, self.chunk_size), (None,) * num_workers)
            for read in iters:
                work.put(read)

            for p in pool:
                p.join()

            q_pbar.put(None)
            proc.join()

            self.logger.info('{}Writing outputs...{}'.format(
                colors.OKBLUE,
                colors.ENDC))

            num_rrna = 0
            num_unknown = 0
            for r1_dict, r2_dict in results:

                if r1_dict[0]:
                    norrna1_fh.write('\n'.join(r1_dict[0]) + '\n')
                    norrna2_fh.write('\n'.join(r2_dict[0]) + '\n')
                if self.rrna is not None and r1_dict[1]:
                    rrna1_fh.write('\n'.join(r1_dict[1]) + '\n')
                    rrna2_fh.write('\n'.join(r2_dict[1]) + '\n')
                    num_rrna += len(r1_dict[1])

                if self.args.ensure == 'both' and r1_dict[-1]:
                    unclf1_fh.write('\n'.join(r1_dict[-1]) + '\n')
                    unclf2_fh.write('\n'.join(r2_dict[-1]) + '\n')
                    num_unknown += len(r1_dict[-1])

                    # del r1_data, r2_data, r1_output, r2_output, r1_batch_labels, r2_batch_labels

            self.logger.info('Done! Detected {}{}{}{} rRNA sequences, discarded {}{}{}{} unclassified sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_rrna,
                colors.ENDC,
                colors.BOLD,
                colors.OKCYAN,
                num_unknown,
                colors.ENDC))

            if self.rrna is not None:
                rrna1_fh.close()
                rrna2_fh.close()

            if self.args.ensure == 'both':
                unclf1_fh.close()
                unclf2_fh.close()

            norrna1_fh.close()
            norrna2_fh.close()

        else:
            input_reads = SeqEncoder.load_reads(*self.input)

            num_seqs = len(input_reads)

            self.num_chunks = math.ceil(
                num_seqs / (self.chunk_size))

            self.logger.info('{}{}{}{} sequences loaded!'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_seqs,
                colors.ENDC))

            if self.rrna is not None:
                self.logger.info('Writing output rRNA sequences into file: {}{}{}'.format(
                    colors.OKBLUE,
                    ", ".join(self.rrna),
                    colors.ENDC))

                rrna_fh = open_for_write(self.rrna[0])

            self.logger.info('Writing output non-rRNA sequences into file: {}{}{}'.format(
                colors.OKBLUE,
                ", ".join(self.output),
                colors.ENDC))

            norrna_fh = open_for_write(self.output[0])

            proc = mp.Process(target=self.listener, args=(q_pbar,))
            proc.start()

            for i in range(num_workers):
                p = mp.Process(target=self.classify_reads,
                               args=(work, results, q_pbar))
                p.start()
                pool.append(p)

            iters = itertools.chain(Predictor.generate_read_chunks(
                input_reads, self.chunk_size), (None,) * num_workers)
            for read in iters:
                work.put(read)

            for p in pool:
                p.join()

            q_pbar.put(None)
            proc.join()

            self.logger.info('{}Writing outputs...{}'.format(
                colors.OKBLUE,
                colors.ENDC))

            num_rrna = 0
            for r_dict in results:

                if r_dict[0]:
                    norrna_fh.write('\n'.join(r_dict[0]) + '\n')
                if self.rrna is not None and r_dict[1]:
                    rrna_fh.write('\n'.join(r_dict[1]) + '\n')
                    num_rrna += len(r_dict[1])

            self.logger.info('Done! Detected {}{}{}{} rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_rrna,
                colors.ENDC
            ))
            if self.rrna is not None:
                rrna_fh.close()
            norrna_fh.close()

    @staticmethod
    def generate_read_chunks(reads, n):
        """Yield successive n-sized chunks from read list."""
        for i in range(0, len(reads), n):
            yield reads[i:i + n]

    @staticmethod
    def generate_paired_read_chunks(reads, n):
        """Yield successive n-sized chunks from paired end read list."""
        r1, r2 = reads
        for i in range(0, len(r1), n):
            yield r1[i:i + n], r2[i:i + n]

    @staticmethod
    def separate_reads(reads, labels):
        reads_dict = defaultdict(list)
        for read, label in zip(reads, labels):

            reads_dict[label].append('\n'.join(read))
        return reads_dict

    def separate_paired_reads(self, r1, r1_labels, r2, r2_labels):
        r1_dict = defaultdict(list)
        r2_dict = defaultdict(list)

        if self.args.ensure == 'rrna':
            for r1, r1_label, r2, r2_label in zip(r1, r1_labels, r2, r2_labels):
                if r1_label == r2_label == 1:
                    final_label = 1
                else:
                    final_label = 0
                r1_dict[final_label].append('\n'.join(r1))
                r2_dict[final_label].append('\n'.join(r2))
        elif self.args.ensure == 'norrna':
            for r1, r1_label, r2, r2_label in zip(r1, r1_labels, r2, r2_labels):
                if r1_label == r2_label == 0:
                    final_label = 0
                else:
                    final_label = 1

                r1_dict[final_label].append('\n'.join(r1))
                r2_dict[final_label].append('\n'.join(r2))
        else:
            for r1, r1_label, r2, r2_label in zip(r1, r1_labels, r2, r2_labels):
                if r1_label == r2_label == 0:
                    final_label = 0
                elif r1_label == r2_label == 1:
                    final_label = 1
                else:
                    final_label = -1

                r1_dict[final_label].append('\n'.join(r1))
                r2_dict[final_label].append('\n'.join(r2))

        return r1_dict, r2_dict

    def classify_reads(self, chunk, out_list, q_pbar):
        while True:
            reads = chunk.get()
            if reads == None:
                return

            input_encoded_reads = np.array([SeqEncoder.encode_variable_len_read(
                read[1], max_len=self.len) for read in reads], dtype=np.float32)

            inputs = {self.model.get_inputs()[0].name: input_encoded_reads}
            outputs = self.model.run(None, inputs)

            out_list.append(Predictor.separate_reads(
                reads, np.argmax(outputs[0], axis=1)))

            q_pbar.put(self.SENTINEL)

    def classify_paired_reads(self, chunk, out_list, q_pbar):
        while True:
            reads = chunk.get()
            if reads == None:
                return

            r1, r2 = reads

            input_encoded_r1 = np.array([SeqEncoder.encode_variable_len_read(
                read[1], max_len=self.len) for read in r1], dtype=np.float32)

            input_encoded_r2 = np.array([SeqEncoder.encode_variable_len_read(
                read[1], max_len=self.len) for read in r2], dtype=np.float32)

            output_labels_r1 = np.argmax(self.model.run(
                None, {self.model.get_inputs()[0].name: input_encoded_r1})[0], axis=1)

            output_labels_r2 = np.argmax(self.model.run(
                None, {self.model.get_inputs()[0].name: input_encoded_r2})[0], axis=1)

            out_list.append(self.separate_paired_reads(
                r1, output_labels_r1, r2, output_labels_r2))
            # print('.', end='')
            q_pbar.put(self.SENTINEL)

    def listener(self, q_pbar):
        pbar = tqdm(total=self.num_chunks)
        for _ in iter(q_pbar.get, None):
            pbar.update()


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='rRNA sequence detector', formatter_class=RawTextHelpFormatter)
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Path of config file')

    args.add_argument('-l', '--len', type=int, required=True,
                      help='Sequencing read length, should be not smaller than 50.')
    args.add_argument('-i', '--input', default=None, type=str, nargs='*', required=True,
                      help='Path of input sequence files (fasta and fastq), the second file will be considered as second end if two files given.')
    args.add_argument('-o', '--output', default=None, type=str, nargs='*', required=True,
                      help='Path of the output sequence files after rRNAs removal (same number of files as input). \n(Note: 2 times slower to write gz files)')
    args.add_argument('-r', '--rrna', default=None, type=str, nargs='*',
                      help='Path of the output sequence file of detected rRNAs (same number of files as input)')
    args.add_argument('-e', '--ensure', default="rrna", type=str, choices=['rrna', 'norrna', 'both'],
                      help='''Only output certain sequences with high confidence 
norrna: output non-rRNAs with high confidence, remove as many rRNAs as possible;
rrna: vice versa, output rRNAs with high confidence;
both: both non-rRNA and rRNA prediction with high confidence.
      (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)''')

    args.add_argument('-t', '--threads', default=10, type=int,
                      help='number of threads to use. (default: 10)')

    args.add_argument('--chunk_size', default=256, type=int,
                      help='chunk_size * threads reads to process per thread.(default: 256) \n{}.'.format(
                          'When chunk_size=1024 and threads=10, each process will load 1024 reads, in total consumming 10G memory'
                      ))

    if not isinstance(args, tuple):
        args = args.parse_args()
    if args.config is None:
        config_file = os.path.join(cd, 'config.json')
    else:
        config_file = args.config
    config = ConfigParser.from_json(config_file)

    os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'

    seq_pred = Predictor(config, args)
    seq_pred.load_model()
    seq_pred.run()
