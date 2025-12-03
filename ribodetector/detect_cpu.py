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
from collections import defaultdict
from ribodetector import __version__

# Timeout for process joins (seconds)
PROCESS_JOIN_TIMEOUT = 30

from argparse import RawTextHelpFormatter
from ribodetector.parse_config import ConfigParser
import ribodetector.data_loader.seq_encoder as SeqEncoder

# Get the directory of the program
cd = os.path.dirname(os.path.abspath(__file__))


def _create_onnx_session(model_file):
    """Create an ONNX inference session with proper settings.

    This is called inside worker processes to ensure each process
    has its own ONNX session (ONNX sessions cannot be shared across
    process boundaries safely).
    """
    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(model_file, so)


def _worker_classify_reads(work_queue, result_container, model_file, seq_len, q_pbar=None):
    """Worker function for classifying reads.

    This runs in a separate process and loads its own ONNX model.
    Using a module-level function instead of a method avoids pickling
    the ONNX session, which doesn't work correctly across process boundaries.

    Args:
        work_queue: Manager Queue of read batches to process (None signals stop)
        result_container: Manager list (use append) or Queue (use put) to store results
        model_file: Path to ONNX model file
        seq_len: Maximum sequence length for encoding
        q_pbar: Optional progress bar queue
    """
    import sys
    # Load model in this worker process
    try:
        model = _create_onnx_session(model_file)
        input_name = model.get_inputs()[0].name
    except Exception as e:
        print(f"Worker failed to load model: {e}", file=sys.stderr, flush=True)
        return

    # Detect result container type - list uses append, queue uses put
    use_append = hasattr(result_container, 'append')

    while True:
        try:
            reads = work_queue.get()
            if reads is None:
                return

            input_encoded_reads = np.array([
                SeqEncoder.encode_variable_len_read(read[1], max_len=seq_len)
                for read in reads
            ], dtype=np.float32)

            outputs = model.run(None, {input_name: input_encoded_reads})

            # Separate reads by predicted label
            reads_dict = defaultdict(list)
            labels = np.argmax(outputs[0], axis=1)
            for read, label in zip(reads, labels):
                reads_dict[label].append('\n'.join(read))

            result = dict(reads_dict)
            if use_append:
                result_container.append(result)
            else:
                result_container.put(result)
            if q_pbar is not None:
                q_pbar.put(1)
        except Exception as e:
            print(f"Worker error: {e}", file=sys.stderr, flush=True)


def _worker_classify_paired_reads(work_queue, result_container, model_file, seq_len, ensure_mode, q_pbar=None):
    """Worker function for classifying paired-end reads.

    This runs in a separate process and loads its own ONNX model.

    Args:
        work_queue: Manager Queue of read pair batches to process (None signals stop)
        result_container: Manager list (use append) or Queue (use put) to store results
        model_file: Path to ONNX model file
        seq_len: Maximum sequence length for encoding
        ensure_mode: The ensure mode for classification
        q_pbar: Optional progress bar queue
    """
    import sys
    # Load model in this worker process
    try:
        model = _create_onnx_session(model_file)
        input_name = model.get_inputs()[0].name
    except Exception as e:
        print(f"Worker failed to load model: {e}", file=sys.stderr, flush=True)
        return

    # Detect result container type - list uses append, queue uses put
    use_append = hasattr(result_container, 'append')

    while True:
        try:
            reads = work_queue.get()
            if reads is None:
                return

            r1, r2 = reads

            input_encoded_r1 = np.array([
                SeqEncoder.encode_variable_len_read(read[1], max_len=seq_len)
                for read in r1
            ], dtype=np.float32)

            input_encoded_r2 = np.array([
                SeqEncoder.encode_variable_len_read(read[1], max_len=seq_len)
                for read in r2
            ], dtype=np.float32)

            output_r1 = model.run(None, {input_name: input_encoded_r1})[0]
            output_r2 = model.run(None, {input_name: input_encoded_r2})[0]

            # Separate reads by predicted labels based on ensure_mode
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

            result = (dict(r1_dict), dict(r2_dict))
            if use_append:
                result_container.append(result)
            else:
                result_container.put(result)
            if q_pbar is not None:
                q_pbar.put(1)
        except Exception as e:
            print(f"Worker error: {e}", file=sys.stderr, flush=True)

## makes the socket addresses extremely random again to esure no address conflicts
multiprocessing.util.abstract_sockets_supported = False


def terminate_process_with_timeout(process, timeout=PROCESS_JOIN_TIMEOUT):
    """Safely terminate a process with timeout.

    Args:
        process: multiprocessing.Process to terminate
        timeout: seconds to wait before force killing

    Returns:
        bool: True if process ended cleanly, False if force killed
    """
    if process is None or not process.is_alive():
        return True

    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        return False
    return True


def cleanup_processes(pool, listener_proc=None, q_pbar=None, logger=None):
    """Clean up all worker processes and listener.

    Args:
        pool: list of worker processes
        listener_proc: optional listener process
        q_pbar: optional progress bar queue
        logger: optional logger for warnings
    """
    # First, try to signal listener to stop by sending None
    if q_pbar is not None:
        try:
            q_pbar.put_nowait(None)
        except Exception:
            pass

    # Terminate all worker processes
    for p in pool:
        if not terminate_process_with_timeout(p):
            if logger:
                logger.warning(f'Worker process {p.pid} had to be force killed')

    # Clean up listener
    if listener_proc is not None:
        if not terminate_process_with_timeout(listener_proc, timeout=5):
            if logger:
                logger.warning(f'Listener process {listener_proc.pid} had to be force killed')


# Global registry for active processes (for signal handler cleanup)
_active_processes = []
_active_listener = None
_active_queue = None


def register_active_processes(pool, listener=None, queue=None):
    """Register processes for signal handler cleanup."""
    global _active_processes, _active_listener, _active_queue
    _active_processes = pool
    _active_listener = listener
    _active_queue = queue


def clear_active_processes():
    """Clear the active process registry."""
    global _active_processes, _active_listener, _active_queue
    _active_processes = []
    _active_listener = None
    _active_queue = None


def graceful_shutdown_handler(signum, frame):
    """Signal handler for graceful shutdown (SIGTERM/SIGINT).

    Ensures all child processes are properly terminated when the main
    process receives a termination signal (e.g., from Docker stop).
    """
    import sys
    global _active_processes, _active_listener, _active_queue

    # Clean up any active processes
    if _active_processes or _active_listener:
        cleanup_processes(_active_processes, _active_listener, _active_queue)
        clear_active_processes()

    # Exit with appropriate code
    sys.exit(128 + signum)


def setup_signal_handlers():
    """Install signal handlers for graceful shutdown."""
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

        # Note: The ONNX model is loaded inside worker processes, not here.
        # This is because ONNX sessions cannot be safely shared across process
        # boundaries when using fork(). Each worker loads its own model.

    def run(self):
        """
        Load data and run the predictor.

        Uses module-level worker functions that load their own ONNX models
        to avoid issues with sharing ONNX sessions across process boundaries.
        """

        num_workers = self.args.threads

        # Manager for multiprocessing - use Manager proxies for reliable IPC
        manager = mp.Manager()
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

            # Start the listener process to monitor the progress
            proc = mp.Process(target=self.listener, args=(q_pbar,))
            proc.start()

            # Start the classification processes using module-level worker function
            # Each worker loads its own ONNX model to avoid fork issues
            for _i in range(num_workers):
                p = mp.Process(
                    target=_worker_classify_paired_reads,
                    args=(work, results, self.model_file, self.len, self.args.ensure, q_pbar)
                )
                p.start()
                pool.append(p)

            # Register processes for signal handler cleanup
            register_active_processes(pool, proc, q_pbar)

            try:
                # Input reads batches
                iters = itertools.chain(Predictor.generate_paired_read_batches(
                    input_reads, self.batch_size), (None,) * num_workers)
                for read in iters:
                    work.put(read)

                # Wait for workers with timeout
                for p in pool:
                    if not terminate_process_with_timeout(p):
                        self.logger.warning(f'Worker process {p.pid} had to be force killed')
            finally:
                # Ensure listener gets the stop signal and is cleaned up
                cleanup_processes(pool, proc, q_pbar, self.logger)
                clear_active_processes()

            self.logger.info('{}Writing outputs...{}'.format(
                colors.OKBLUE,
                colors.ENDC))

            for r1_dict, r2_dict in results:
                # Load the prediction results and split the input reads accordingly

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

            self.logger.info('Writing output non-rRNA sequences into file: {}{}{}'.format(
                colors.OKBLUE,
                ", ".join(self.output),
                colors.ENDC))

            norrna_fh = open_for_write(self.output[0])

            # Start the listener process to monitor the progress
            proc = mp.Process(target=self.listener, args=(q_pbar,))
            proc.start()

            # Start the classification processes using module-level worker function
            # Each worker loads its own ONNX model to avoid fork issues
            for _i in range(num_workers):
                p = mp.Process(
                    target=_worker_classify_reads,
                    args=(work, results, self.model_file, self.len, q_pbar)
                )
                p.start()
                pool.append(p)

            # Register processes for signal handler cleanup
            register_active_processes(pool, proc, q_pbar)

            try:
                iters = itertools.chain(Predictor.generate_read_batches(
                    input_reads, self.batch_size), (None,) * num_workers)
                for read in iters:
                    work.put(read)

                # Wait for workers with timeout
                for p in pool:
                    if not terminate_process_with_timeout(p):
                        self.logger.warning(f'Worker process {p.pid} had to be force killed')
            finally:
                # Ensure listener gets the stop signal and is cleaned up
                cleanup_processes(pool, proc, q_pbar, self.logger)
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
                # Use regular queues instead of Manager proxies for reliability
                work_queue = mp.Queue(num_workers * 2)
                result_queue = mp.Queue()

                pool = []

                # Start the classification processes using module-level worker function
                for _i in range(num_workers):
                    p = mp.Process(
                        target=_worker_classify_paired_reads,
                        args=(work_queue, result_queue, self.model_file, self.len, self.args.ensure)
                    )
                    p.start()
                    pool.append(p)

                # Register processes for signal handler cleanup
                register_active_processes(pool)

                try:
                    # Send work batches to workers
                    batches_sent = 0
                    for batch in Predictor.generate_paired_read_batches(chunk, self.batch_size):
                        work_queue.put(batch)
                        batches_sent += 1

                    # Send stop signals
                    for _ in range(num_workers):
                        work_queue.put(None)

                    # Collect results
                    results_received = 0
                    while results_received < batches_sent:
                        try:
                            result = result_queue.get(timeout=60)
                            r1_dict, r2_dict = result
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

                            results_received += 1
                        except Exception as e:
                            self.logger.warning(f'Error getting result: {e}')
                            break

                    # Wait for workers to finish
                    for p in pool:
                        if not terminate_process_with_timeout(p):
                            self.logger.warning(f'Worker process {p.pid} had to be force killed')
                finally:
                    # Ensure all processes are cleaned up even on error
                    cleanup_processes(pool, logger=self.logger)
                    clear_active_processes()

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

                # Use regular queues instead of Manager proxies for reliability
                work_queue = mp.Queue(num_workers * 2)
                result_queue = mp.Queue()

                pool = []

                for _i in range(num_workers):
                    p = mp.Process(
                        target=_worker_classify_reads,
                        args=(work_queue, result_queue, self.model_file, self.len)
                    )
                    p.start()
                    pool.append(p)

                # Register processes for signal handler cleanup
                register_active_processes(pool)

                try:
                    # Send work batches to workers
                    batches_sent = 0
                    for batch in Predictor.generate_read_batches(chunk, self.batch_size):
                        work_queue.put(batch)
                        batches_sent += 1

                    # Send stop signals
                    for _ in range(num_workers):
                        work_queue.put(None)

                    # Collect results
                    results_received = 0
                    while results_received < batches_sent:
                        try:
                            result = result_queue.get(timeout=60)
                            r_dict = result
                            num_nonrrna += len(r_dict.get(0, []))
                            num_rrna += len(r_dict.get(1, []))

                            if r_dict.get(0):
                                norrna_fh.write('\n'.join(r_dict[0]) + '\n')
                            if self.rrna is not None and r_dict.get(1):
                                rrna_fh.write('\n'.join(r_dict[1]) + '\n')

                            results_received += 1
                        except Exception as e:
                            self.logger.warning(f'Error getting result: {e}')
                            break

                    # Wait for workers to finish
                    for p in pool:
                        if not terminate_process_with_timeout(p):
                            self.logger.warning(f'Worker process {p.pid} had to be force killed')
                finally:
                    # Ensure all processes are cleaned up even on error
                    cleanup_processes(pool, logger=self.logger)
                    clear_active_processes()

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
        """Progress bar listener with timeout protection.

        Uses a timeout on queue gets to prevent hanging indefinitely
        if the main process crashes without sending the stop signal.
        """
        from queue import Empty
        pbar = tqdm(total=self.num_batches)
        try:
            while True:
                try:
                    item = q_pbar.get(timeout=60)  # 60 second timeout
                    if item is None:
                        break
                    pbar.update()
                except Empty:
                    # Check if we should continue waiting
                    # After timeout, continue waiting unless parent process died
                    continue
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

    # Set up signal handlers for graceful shutdown (important for Docker)
    setup_signal_handlers()

    if platform.system() == 'Darwin':
        mp.set_start_method('fork')
    seq_pred = Predictor(config, args)
    seq_pred.load_model()

    seq_pred.detect()


if __name__ == '__main__':
    main()
