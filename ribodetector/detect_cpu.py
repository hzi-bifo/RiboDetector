#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
File: detect_cpu.py
Created Date: January 1st 2020
Author: ZL Deng <dawnmsg(at)gmail.com>
---------------------------------------
Last Modified: 7th March 2022 12:02:19 pm
'''

import os
import math
import gzip
import signal
import argparse
import platform
import itertools
import onnxruntime
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import multiprocessing.util
from queue import Empty, Full
from collections import defaultdict
from ribodetector import __version__

from argparse import RawTextHelpFormatter
from ribodetector.parse_config import ConfigParser
import ribodetector.data_loader.seq_encoder as SeqEncoder

# Get the directory of the program
cd = os.path.dirname(os.path.abspath(__file__))

## makes the socket addresses extremely random again to esure no address conflicts
multiprocessing.util.abstract_sockets_supported = False


def _create_onnx_session(model_file):
    """Create an ONNX session inside a worker process."""
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(model_file, so)


def _worker_error_msg(worker_name, exc):
    return '{} failed ({}): {}'.format(worker_name, exc.__class__.__name__, exc)


def _worker_classify_reads(work_queue, out_list, error_queue, model_file, seq_len, q_pbar=None):
    """Classify single-end reads in worker process."""
    try:
        model = _create_onnx_session(model_file)
        input_name = model.get_inputs()[0].name
    except Exception as exc:
        error_queue.put(_worker_error_msg('worker init', exc))
        return

    while True:
        reads = work_queue.get()
        if reads is None:
            return
        try:
            input_encoded_reads = np.array([SeqEncoder.encode_variable_len_read(
                read[1], max_len=seq_len) for read in reads], dtype=np.float32)
            outputs = model.run(None, {input_name: input_encoded_reads})
            labels = np.argmax(outputs[0], axis=1)
            reads_dict = defaultdict(list)
            for read, label in zip(reads, labels):
                reads_dict[label].append('\n'.join(read))
            out_list.append(dict(reads_dict))
            if q_pbar is not None:
                q_pbar.put(1)
        except Exception as exc:
            error_queue.put(_worker_error_msg('worker classify batch', exc))
            return


def _worker_classify_paired_reads(work_queue, out_list, error_queue, model_file, seq_len, ensure_mode, q_pbar=None):
    """Classify paired-end reads in worker process."""
    try:
        model = _create_onnx_session(model_file)
        input_name = model.get_inputs()[0].name
    except Exception as exc:
        error_queue.put(_worker_error_msg('worker init', exc))
        return

    while True:
        reads = work_queue.get()
        if reads is None:
            return
        try:
            r1, r2 = reads

            input_encoded_r1 = np.array([SeqEncoder.encode_variable_len_read(
                read[1], max_len=seq_len) for read in r1], dtype=np.float32)
            input_encoded_r2 = np.array([SeqEncoder.encode_variable_len_read(
                read[1], max_len=seq_len) for read in r2], dtype=np.float32)

            output_r1 = model.run(None, {input_name: input_encoded_r1})[0]
            output_r2 = model.run(None, {input_name: input_encoded_r2})[0]

            r1_dict = defaultdict(list)
            r2_dict = defaultdict(list)

            if ensure_mode == 'rrna':
                r1_labels = np.argmax(output_r1, axis=1)
                r2_labels = np.argmax(output_r2, axis=1)
                for r1_read, r1_label, r2_read, r2_label in zip(r1, r1_labels, r2, r2_labels):
                    final_label = 1 if r1_label == r2_label == 1 else 0
                    r1_dict[final_label].append('\n'.join(r1_read))
                    r2_dict[final_label].append('\n'.join(r2_read))
            elif ensure_mode == 'norrna':
                r1_labels = np.argmax(output_r1, axis=1)
                r2_labels = np.argmax(output_r2, axis=1)
                for r1_read, r1_label, r2_read, r2_label in zip(r1, r1_labels, r2, r2_labels):
                    final_label = 0 if r1_label == r2_label == 0 else 1
                    r1_dict[final_label].append('\n'.join(r1_read))
                    r2_dict[final_label].append('\n'.join(r2_read))
            elif ensure_mode == 'both':
                r1_labels = np.argmax(output_r1, axis=1)
                r2_labels = np.argmax(output_r2, axis=1)
                for r1_read, r1_label, r2_read, r2_label in zip(r1, r1_labels, r2, r2_labels):
                    if r1_label == r2_label == 0:
                        final_label = 0
                    elif r1_label == r2_label == 1:
                        final_label = 1
                    else:
                        final_label = -1
                    r1_dict[final_label].append('\n'.join(r1_read))
                    r2_dict[final_label].append('\n'.join(r2_read))
            else:
                final_labels = np.argmax(output_r1 + output_r2, axis=1)
                for r1_read, r2_read, final_label in zip(r1, r2, final_labels):
                    r1_dict[final_label].append('\n'.join(r1_read))
                    r2_dict[final_label].append('\n'.join(r2_read))

            out_list.append((dict(r1_dict), dict(r2_dict)))
            if q_pbar is not None:
                q_pbar.put(1)
        except Exception as exc:
            error_queue.put(_worker_error_msg('worker classify batch', exc))
            return


def _drain_errors(error_queue):
    errors = []
    while True:
        try:
            errors.append(error_queue.get_nowait())
        except Empty:
            return errors
        except Exception:
            return errors


def _raise_worker_errors(error_queue):
    errors = _drain_errors(error_queue)
    if errors:
        raise RuntimeError(errors[0])


def _put_with_worker_error_check(work, item, error_queue, pool=None):
    while True:
        _raise_worker_errors(error_queue)
        if pool is not None:
            _raise_if_worker_exit_failed(pool)
        try:
            work.put(item, timeout=1)
            return
        except Full:
            continue


def _raise_if_worker_exit_failed(pool):
    failed = [(p.pid, p.exitcode) for p in pool if p.exitcode not in (0, None)]
    if failed:
        raise RuntimeError('Worker process exited abnormally: {}'.format(failed))


def _terminate_process(proc, timeout=5):
    if proc is None:
        return
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=1)


def cleanup_processes(pool, listener_proc=None, q_pbar=None):
    if q_pbar is not None:
        try:
            q_pbar.put_nowait(None)
        except Exception:
            pass
    for p in pool:
        _terminate_process(p)
    _terminate_process(listener_proc)


_active_pool = []
_active_listener = None
_active_q_pbar = None


def register_active_processes(pool, listener=None, q_pbar=None):
    global _active_pool, _active_listener, _active_q_pbar
    _active_pool = pool
    _active_listener = listener
    _active_q_pbar = q_pbar


def clear_active_processes():
    global _active_pool, _active_listener, _active_q_pbar
    _active_pool = []
    _active_listener = None
    _active_q_pbar = None


def graceful_shutdown_handler(signum, frame):
    cleanup_processes(_active_pool, _active_listener, _active_q_pbar)
    clear_active_processes()
    raise SystemExit(128 + signum)


def setup_signal_handlers():
    signal.signal(signal.SIGTERM, graceful_shutdown_handler)
    signal.signal(signal.SIGINT, graceful_shutdown_handler)

class Predictor:
    """
    Main class of predictor for rRNA, non-rRNA sequences
    """

    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.logger = config.get_logger('predict', 1, self.args.log)
        self.chunk_size = self.args.chunk_size

    def load_model(self):
        """Load the right model file for classification 

        Raises:
            RuntimeError: raise error if input sequence length <40
        """

        self.len = self.args.len

        if self.len < 40:
            #             self.logger.error('{}Sequence length is too short to classify!{}'.format(
            #                 colors.FAIL,
            #                 colors.ENDC))
            #             raise RuntimeError(
            #                 "Sequence length must be set to larger than 40.")
            self.logger.info(
                'The accuracy will drop with reads shorter than 40.')

        # High recall model if ensure non-rRNA
        if self.args.ensure == 'norrna':
            model_file_ext = 'recall'
        else:
            model_file_ext = 'mcc'

        self.model_file = os.path.join(
            cd, self.config['state_file'][model_file_ext]).replace('.pth', '.onnx')

        # self.logger.info('Using high {} model file: {}{}{}{} on CPU'.format(model_file_ext.upper(),
        #                                                                     colors.BOLD,
        #                                                                     colors.OKCYAN,
        #                                                                     self.model_file,
        #                                                                     colors.ENDC))
        self.logger.info('Using high {} model'.format(model_file_ext.upper()))
        
        self.logger.info('Log file: {}'.format(
            self.args.log
            ))

    def run(self):
        """
        Load data and run the predictor
        """

        num_workers = self.args.threads

        # Manager for multiprocessing
        manager = mp.Manager()

        # List to store prediction results
        results = manager.list()
        work = manager.Queue(num_workers)

        pool = []

        # queue for progressbar signal
        q_pbar = mp.Queue()
        
        num_nonrrna = 0
        num_rrna = 0
        
        if self.is_paired:
            # Load paired end read files with multiprocessing
            with mp.Pool(2) as p:
                input_reads = p.map(SeqEncoder.load_reads, self.input)

            num_seqs = len(input_reads[0])

            self.num_batches = math.ceil(
                num_seqs / (self.batch_size))

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

                num_unknown = 0

            error_queue = manager.Queue()

            # Start the listener process to monitor the progress
            proc = mp.Process(target=self.listener, args=(q_pbar,))
            proc.start()

            # Start the classification processes
            for _i in range(num_workers):
                p = mp.Process(
                    target=_worker_classify_paired_reads,
                    args=(work, results, error_queue, self.model_file, self.len, self.args.ensure, q_pbar)
                )
                p.start()
                pool.append(p)

            register_active_processes(pool, proc, q_pbar)

            try:
                # Input read batches
                for read in Predictor.generate_paired_read_batches(input_reads, self.batch_size):
                    _put_with_worker_error_check(work, read, error_queue, pool)
                for _ in range(num_workers):
                    _put_with_worker_error_check(work, None, error_queue, pool)

                for p in pool:
                    p.join()
                _raise_worker_errors(error_queue)
                _raise_if_worker_exit_failed(pool)
            except Exception:
                cleanup_processes(pool, proc, q_pbar)
                clear_active_processes()
                raise
            else:
                q_pbar.put(None)
                proc.join()
                clear_active_processes()

            self.logger.info('{}Writing outputs...{}'.format(
                colors.OKBLUE,
                colors.ENDC))

            for r1_dict, r2_dict in results:
                # Load the prediciton results and split the input reads accordingly
                
                num_nonrrna += len(r1_dict.get(0, []))
                num_rrna += len(r1_dict.get(1, []))

                if r1_dict.get(0):
                    norrna1_fh.write('\n'.join(r1_dict[0]) + '\n')
                    norrna2_fh.write('\n'.join(r2_dict[0]) + '\n')
                if self.rrna is not None and r1_dict.get(1):
                    rrna1_fh.write('\n'.join(r1_dict[1]) + '\n')
                    rrna2_fh.write('\n'.join(r2_dict[1]) + '\n')

                if self.args.ensure == 'both' and r1_dict.get(-1):
                    unclf1_fh.write('\n'.join(r1_dict[-1]) + '\n')
                    unclf2_fh.write('\n'.join(r2_dict[-1]) + '\n')
                    num_unknown += len(r1_dict[-1])
            
            self.logger.info('Processed {}{}{}{} sequences in total'.format(
                        colors.BOLD,
                        colors.OKCYAN,
                        num_seqs,
                        colors.ENDC))

            self.logger.info('Detected {}{}{}{} non-rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_nonrrna,
                colors.ENDC
            ))

            self.logger.info('Detected {}{}{}{} rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_rrna,
                colors.ENDC
            ))

            if self.rrna is not None:
                rrna1_fh.close()
                rrna2_fh.close()

            if self.args.ensure == 'both':
                self.logger.info('Discarded {}{}{}{} unclassified sequences'.format(
                    colors.BOLD,
                    colors.OKCYAN,
                    num_unknown,
                    colors.ENDC))

                unclf1_fh.close()
                unclf2_fh.close()

            norrna1_fh.close()
            norrna2_fh.close()

        else:
            input_reads = SeqEncoder.load_reads(*self.input)

            num_seqs = len(input_reads)

            self.num_batches = math.ceil(
                num_seqs / (self.batch_size))

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
#                num_rrna = 0

            self.logger.info('Writing output non-rRNA sequences into file: {}{}{}'.format(
                colors.OKBLUE,
                ", ".join(self.output),
                colors.ENDC))

            norrna_fh = open_for_write(self.output[0])

            error_queue = manager.Queue()

            proc = mp.Process(target=self.listener, args=(q_pbar,))
            proc.start()

            for _i in range(num_workers):
                p = mp.Process(
                    target=_worker_classify_reads,
                    args=(work, results, error_queue, self.model_file, self.len, q_pbar)
                )
                p.start()
                pool.append(p)

            register_active_processes(pool, proc, q_pbar)

            try:
                for read in Predictor.generate_read_batches(input_reads, self.batch_size):
                    _put_with_worker_error_check(work, read, error_queue, pool)
                for _ in range(num_workers):
                    _put_with_worker_error_check(work, None, error_queue, pool)

                for p in pool:
                    p.join()
                _raise_worker_errors(error_queue)
                _raise_if_worker_exit_failed(pool)
            except Exception:
                cleanup_processes(pool, proc, q_pbar)
                clear_active_processes()
                raise
            else:
                q_pbar.put(None)
                proc.join()
                clear_active_processes()

            self.logger.info('{}Writing outputs...{}'.format(
                colors.OKBLUE,
                colors.ENDC))

            for r_dict in results:

                num_nonrrna += len(r_dict.get(0, []))
                num_rrna += len(r_dict.get(1, []))
                if r_dict.get(0):
                    norrna_fh.write('\n'.join(r_dict[0]) + '\n')
                if self.rrna is not None and r_dict.get(1):
                    rrna_fh.write('\n'.join(r_dict[1]) + '\n')

            self.logger.info('Processed {}{}{}{} sequences in total'.format(
                        colors.BOLD,
                        colors.OKCYAN,
                        num_seqs,
                        colors.ENDC))
            
            self.logger.info('Detected {}{}{}{} non-rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_nonrrna,
                colors.ENDC
            ))

            self.logger.info('Detected {}{}{}{} rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_rrna,
                colors.ENDC
            ))

            if self.rrna is not None:
                rrna_fh.close()
            norrna_fh.close()

    def run_with_chunks(self):
        """
        Load data with chunks and run the predictor
        """

        num_workers = self.args.threads

        read_chunk_size = self.batch_size * self.chunk_size

        num_read = 0
        num_nonrrna = 0
        num_rrna = 0

        if self.is_paired:
            self.logger.info('Classify paired-end reads with chunk size {}{}{}'.format(
                colors.BOLD,
                self.chunk_size,
                colors.ENDC))
            if self.rrna is not None:
                self.logger.info('Writing output rRNA sequences into file: {}{}{}'.format(
                    colors.OKBLUE,
                    ", ".join(self.rrna),
                    colors.ENDC))
                rrna1_fh = open_for_write(self.rrna[0])
                rrna2_fh = open_for_write(self.rrna[1])
                # num_rrna = 0

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

                num_unknown = 0

            # num_read = 0

            # Load paired end reads with chunks
            for chunk in SeqEncoder.get_pairedread_chunks(*self.input,
                                                          chunk_size=read_chunk_size):
                # Manager for multiprocessing
                manager = mp.Manager()

                # List to store prediction results
                results = manager.list()
                work = manager.Queue(num_workers)
                error_queue = manager.Queue()

                pool = []

                # Start the classification processes
                for _i in range(num_workers):
                    p = mp.Process(
                        target=_worker_classify_paired_reads,
                        args=(work, results, error_queue, self.model_file, self.len, self.args.ensure)
                    )
                    p.start()
                    pool.append(p)

                register_active_processes(pool)

                try:
                    # Input reads batches for each chunk
                    for read in Predictor.generate_paired_read_batches(chunk, self.batch_size):
                        _put_with_worker_error_check(work, read, error_queue, pool)
                    for _ in range(num_workers):
                        _put_with_worker_error_check(work, None, error_queue, pool)

                    for p in pool:
                        p.join()
                    _raise_worker_errors(error_queue)
                    _raise_if_worker_exit_failed(pool)
                except Exception:
                    cleanup_processes(pool)
                    clear_active_processes()
                    raise
                else:
                    clear_active_processes()

                for r1_dict, r2_dict in results:
                    # Load the prediciton results and split the input reads accordingly

                    num_nonrrna += len(r1_dict.get(0, []))
                    num_rrna += len(r1_dict.get(1, []))
                    
                    if r1_dict.get(0):
                        norrna1_fh.write('\n'.join(r1_dict[0]) + '\n')
                        norrna2_fh.write('\n'.join(r2_dict[0]) + '\n')
                    if self.rrna is not None and r1_dict.get(1):
                        rrna1_fh.write('\n'.join(r1_dict[1]) + '\n')
                        rrna2_fh.write('\n'.join(r2_dict[1]) + '\n')

                    if self.args.ensure == 'both' and r1_dict.get(-1):
                        unclf1_fh.write('\n'.join(r1_dict[-1]) + '\n')
                        unclf2_fh.write('\n'.join(r2_dict[-1]) + '\n')
                        num_unknown += len(r1_dict[-1])
                num_read += len(chunk[0])

                self.logger.info('{}{}{} sequences finished!'.format(
                    colors.OKGREEN,
                    num_read,
                    colors.ENDC))

            self.logger.info('Processed {}{}{}{} sequences in total'.format(
                        colors.BOLD,
                        colors.OKCYAN,
                        num_read,
                        colors.ENDC))
            
            self.logger.info('Detected {}{}{}{} non-rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_nonrrna,
                colors.ENDC
            ))
            
            self.logger.info('Detected {}{}{}{} rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_rrna,
                colors.ENDC
            ))

            if self.rrna is not None:
                rrna1_fh.close()
                rrna2_fh.close()

            if self.args.ensure == 'both':
                self.logger.info('Discarded {}{}{}{} unclassified sequences'.format(
                    colors.BOLD,
                    colors.OKCYAN,
                    num_unknown,
                    colors.ENDC))

                unclf1_fh.close()
                unclf2_fh.close()

            norrna1_fh.close()
            norrna2_fh.close()

        else:
            # num_read = 0
            self.logger.info('Classify reads with chunk size {}{}{}'.format(
                colors.BOLD,
                self.chunk_size,
                colors.ENDC))

            if self.rrna is not None:
                self.logger.info('Writing output rRNA sequences into file: {}{}{}'.format(
                    colors.OKBLUE,
                    ", ".join(self.rrna),
                    colors.ENDC))

                rrna_fh = open_for_write(self.rrna[0])
                # num_rrna = 0

            self.logger.info('Writing output non-rRNA sequences into file: {}{}{}'.format(
                colors.OKBLUE,
                ", ".join(self.output),
                colors.ENDC))

            norrna_fh = open_for_write(self.output[0])

            for chunk in SeqEncoder.get_seq_chunks(*self.input,
                                                   chunk_size=read_chunk_size):

                # Manager for multiprocessing
                manager = mp.Manager()

                # List to store prediction results
                results = manager.list()
                work = manager.Queue(num_workers)
                error_queue = manager.Queue()

                pool = []

                for _i in range(num_workers):
                    p = mp.Process(
                        target=_worker_classify_reads,
                        args=(work, results, error_queue, self.model_file, self.len)
                    )
                    p.start()
                    pool.append(p)

                register_active_processes(pool)

                try:
                    for read in Predictor.generate_read_batches(chunk, self.batch_size):
                        _put_with_worker_error_check(work, read, error_queue, pool)
                    for _ in range(num_workers):
                        _put_with_worker_error_check(work, None, error_queue, pool)

                    for p in pool:
                        p.join()
                    _raise_worker_errors(error_queue)
                    _raise_if_worker_exit_failed(pool)
                except Exception:
                    cleanup_processes(pool)
                    clear_active_processes()
                    raise
                else:
                    clear_active_processes()

                for r_dict in results:
                    
                    num_nonrrna += len(r_dict.get(0, []))
                    num_rrna += len(r_dict.get(1, []))

                    if r_dict.get(0):
                        norrna_fh.write('\n'.join(r_dict[0]) + '\n')
                    if self.rrna is not None and r_dict.get(1):
                        rrna_fh.write('\n'.join(r_dict[1]) + '\n')

                num_read += len(chunk)

                self.logger.info('{}{}{} sequences finished!'.format(
                    colors.OKGREEN,
                    num_read,
                    colors.ENDC))

            self.logger.info('Processed {}{}{}{} sequences in total'.format(
                        colors.BOLD,
                        colors.OKCYAN,
                        num_read,
                        colors.ENDC))
            
            self.logger.info('Detected {}{}{}{} non-rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_nonrrna,
                colors.ENDC
            ))

            self.logger.info('Detected {}{}{}{} rRNA sequences'.format(
                colors.BOLD,
                colors.OKCYAN,
                num_rrna,
                colors.ENDC
            ))
            
            if self.rrna is not None:
                rrna_fh.close()
            norrna_fh.close()

    def detect(self):
        """Wrapper for all steps

        Raises:
            RuntimeError: raise error if the number of input read files and output files are invalid
        """
        self.input = self.args.input
        self.output = self.args.output
        self.rrna = self.args.rrna

        num_inputs = len(self.input)
        num_rrna_outputs = None if self.rrna is None else len(self.rrna)
        num_norrna_outputs = len(self.output)
        if num_inputs != num_norrna_outputs or num_inputs > 2:
            self.logger.error('{}The number of input and output sequence files is invalid!{}'.format(
                colors.FAIL,
                colors.ENDC))
            raise RuntimeError(
                "Input or output should have no more than two files and they should have the same number of files.")
        if num_rrna_outputs is not None and num_rrna_outputs != num_inputs:
            self.logger.error('{}The number of output rRNA sequence files is invalid!{}'.format(
                colors.FAIL,
                colors.ENDC))
            raise RuntimeError(
                "Ouput rRNA should have no more than two files and they should the same number with input files.")

        self.batch_size = 1024
        self.is_paired = (num_inputs == 2)

        if self.chunk_size is None:
            self.run()
        else:
            self.run_with_chunks()

    @staticmethod
    def generate_read_batches(reads, n):
        """Yield successive n-sized batches from read list."""
        for i in range(0, len(reads), n):
            yield reads[i:i + n]

    @staticmethod
    def generate_paired_read_batches(reads, n):
        """Yield successive n-sized batches from paired end read list."""
        r1, r2 = reads
        for i in range(0, len(r1), n):
            yield r1[i:i + n], r2[i:i + n]

    def listener(self, q_pbar):
        pbar = tqdm(total=self.num_batches)
        try:
            for _ in iter(q_pbar.get, None):
                pbar.update()
        finally:
            pbar.close()


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
    """
    Define the color of logger text
    """

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


def main():
    args = argparse.ArgumentParser(
        description='rRNA sequence detector', formatter_class=RawTextHelpFormatter)
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Path of config file')

    args.add_argument('-l', '--len', type=int, required=True,
                      help='Sequencing read length. Note: the accuracy reduces for reads shorter than 40.')
    args.add_argument('-i', '--input', default=None, type=str, nargs='*', required=True,
                      help='Path of input sequence files (fasta and fastq), the second file will be considered as second end if two files given.')
    args.add_argument('-o', '--output', default=None, type=str, nargs='*', required=True,
                      help='Path of the output sequence files after rRNAs removal (same number of files as input). \n(Note: 2 times slower to write gz files)')
    args.add_argument('-r', '--rrna', default=None, type=str, nargs='*',
                      help='Path of the output sequence file of detected rRNAs (same number of files as input)')
    args.add_argument('-e', '--ensure', default="none", type=str, choices=['rrna', 'norrna', 'both', 'none'],
                      help='''Ensure which classificaion has high confidence for paired end reads.
norrna: output only high confident non-rRNAs, the rest are clasified as rRNAs;
rrna: vice versa, only high confident rRNAs are classified as rRNA and the rest output as non-rRNAs;
both: both non-rRNA and rRNA prediction with high confidence;
none: give label based on the mean probability of read pair.
      (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)''')

    args.add_argument('-t', '--threads', default=20, type=int,
                      help='Number of threads to use. (default: 20)')
    args.add_argument('-s', '--seed', default=None, type=int,
                      help='Random seed.')
    args.add_argument('--chunk_size', default=None, type=int,
                      help='chunk_size * 1024 reads to load each time. \n{}.'.format(
                          'When chunk_size=1000 and threads=20, consumming ~20G memory, better to be multiples of the number of threads.'))
    args.add_argument('--log', default=None, type=str,
                      help='Log file name')
    args.add_argument('-v', '--version', action='version',
                      version='%(prog)s {version}'.format(version=__version__))

    if not isinstance(args, tuple):
        args = args.parse_args()
    if args.config is None:
        config_file = os.path.join(cd, 'config.json')
    else:
        config_file = args.config
    config = ConfigParser.from_json(config_file)
    
    if isinstance(args.seed, int):
        onnxruntime.set_seed(args.seed)

    os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'

    setup_signal_handlers()

    if platform.system() == 'Darwin':
        mp.set_start_method('fork')
    seq_pred = Predictor(config, args)
    seq_pred.load_model()

    seq_pred.detect()


if __name__ == '__main__':
    main()
