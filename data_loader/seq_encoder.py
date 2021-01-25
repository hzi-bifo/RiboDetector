# import itertools
import gzip
from Bio.Seq import Seq
from pathlib import Path
from functools import partial
from itertools import islice
from mimetypes import guess_type
from data_loader.fastx_parser import seq_parser


BASE_DICT = {"A": (1, 0, 0, 0),
             "C": (0, 1, 0, 0),
             "G": (0, 0, 1, 0),
             "T": (0, 0, 0, 1),
             "U": (0, 0, 0, 1)
             }

ZERO_LIST = (0, 0, 0, 0)


def get_seq_format(seq_file):
    fa_exts = [".fasta", ".fa", ".fna", ".fas"]
    fq_exts = [".fq", ".fastq"]
    encoding = guess_type(seq_file)[1]  # uses file extension
    if encoding is None:
        encoding = ""
    elif encoding == "gzip":
        encoding = "gz"
    else:
        raise ValueError('Unknown file encoding: "{}"'.format(encoding))
    seq_filename = Path(
        seq_file).stem if encoding == 'gz' else Path(seq_file).name
    seq_file_ext = Path(seq_filename).suffix
    if seq_file_ext not in (fa_exts + fq_exts):
        raise ValueError("""Unknown extension {}. Only fastq and fasta sequence formats are supported.
And the file must end with one of ".fasta", ".fa", ".fna", ".fas", ".fq", ".fastq"
and followed by ".gz" or ".gzip" if they are gzipped.""".format(seq_file_ext))
    seq_format = "fa" + encoding if seq_file_ext in fa_exts else "fq" + encoding
    return seq_format


def load_seqs(seq_file, label, step_size):
    seq_step_list = []
    seq_format = get_seq_format(seq_file)
    _open = partial(gzip.open, mode='rt') if seq_format.endswith(
        "gz") else open
    seq_type = "fasta" if seq_format.startswith("fa") else "fastq"
    with _open(seq_file) as fh:
        for record in seq_parser(fh, seq_type):
            seq = record[1]
            seq_step_list.extend([(label, seq, step_size),
                                  (label, str(Seq(seq).reverse_complement()), step_size)])
    return seq_step_list


def load_reads(seq_file, label=None, max_len=100):
    read_list = []
    seq_format = get_seq_format(seq_file)
    _open = partial(gzip.open, mode='rt') if seq_format.endswith(
        "gz") else open
    seq_type = "fasta" if seq_format.startswith("fa") else "fastq"
    if label is None:
        with _open(seq_file) as fh:
            for record in seq_parser(fh, seq_type):
                read_list.append(record)
    else:
        with _open(seq_file) as fh:
            for record in seq_parser(fh, seq_type):
                seq = record[1]
                read, rc_read = get_read_rc_with_maxlen(seq, max_len=100)
                read_list.extend([(label, read), (label, rc_read)])
    return read_list


def get_seq_chunks(seq_file, chunk_size=1048576):
    seq_format = get_seq_format(seq_file)
    _open = partial(gzip.open, mode='rt') if seq_format.endswith(
        "gz") else open
    seq_type = "fasta" if seq_format.startswith("fa") else "fastq"
    with _open(seq_file) as fh:
        seq_iterator = seq_parser(fh, seq_type)
        while True:
            seqs_chunk = list(islice(seq_iterator, chunk_size))
            if seqs_chunk:
                yield seqs_chunk
            else:
                break


def get_pairedread_chunks(r1_seq_file, r2_seq_file, chunk_size=1048576):
    for r1_chunk, r2_chunk in zip(get_seq_chunks(r1_seq_file, chunk_size), get_seq_chunks(r2_seq_file, chunk_size)):
        yield r1_chunk, r2_chunk


def get_read_rc_with_maxlen(seq, max_len=100):
    read = rc_read = 'N' * max_len
    seq_len = len(seq)
    if seq_len >= max_len:
        start = (seq_len - max_len) // 2
        end = max_len + start
        read = seq[start:end]
        rc_read = str(Seq(read).reverse_complement())
    else:
        missing_len = max_len - seq_len
        read = seq + 'N' * missing_len
        rc_read = str(Seq(seq).reverse_complement()) + 'N' * missing_len

    return read, rc_read


def get_read_with_maxlen(seq, max_len=100):
    read = 'N' * max_len
    seq_len = len(seq)
    if seq_len > max_len:
        start = (seq_len - max_len) // 2
        end = max_len + start
        read = seq[start:end]
    elif seq_len == max_len:
        read = seq
    else:
        missing_len = max_len - seq_len
        read = seq + 'N' * missing_len
    return read


def encode_read(read):
    return [BASE_DICT.get(base, ZERO_LIST) for base in read]


def encode_variable_len_read(read, max_len=100):

    read_len = len(read)
    if read_len >= max_len:
        # start = (read_len - max_len) // 2
        # end = max_len + start
        # encoded_read = [BASE_DICT.get(base, ZERO_LIST) for base in read[start:end]]
        encoded_read = [BASE_DICT.get(base, ZERO_LIST)
                        for base in read[:max_len]]
    # elif read_len == max_len:
    #     encoded_read = [BASE_DICT.get(base, ZERO_LIST) for base in read]
    else:
        encoded_read = [ZERO_LIST] * max_len
        encoded_read[:read_len] = [BASE_DICT.get(
            base, ZERO_LIST) for base in read]
    return encoded_read


def encode_seq_reads(seq, step_size, max_len=100):
    seq_len = len(seq)
    seq_feature = [BASE_DICT.get(base, ZERO_LIST) for base in seq]
    encoded_read_list = []
    for i in range(0, seq_len, step_size):
        encoded_read = [ZERO_LIST] * max_len

        if seq_len >= i + max_len:
            encoded_read_list.append(seq_feature[i:i + max_len])
        else:
            if seq_len > i + max_len / 2:
                encoded_read[:seq_len - i] = seq_feature[i:]
                encoded_read_list.append(encoded_read)
            break
    return encoded_read_list
