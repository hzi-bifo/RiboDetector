# from data_loader.fastx_parser import seq_parser
from torch.utils.data import Dataset, IterableDataset, get_worker_info

# Single end sequence data, and long sequence data


class SeqData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Data is paired end read list with two items
class PairedReadData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        r1 = self.data[0][index]
        r2 = self.data[1][index]
        return r1, r2

    def __len__(self):
        return len(self.data[0])


class IterablePairedReadData(IterableDataset):
    '''
    @params:
        r1, r2: read end pairs
    '''

    def __init__(self, r1, r2, batch_size):
        # super(IterablePairedReadData).__init__()
        self.r1 = r1
        self.r2 = r2
        self.batch_size = batch_size

    def get_stream(self):
        for (r1_read, r2_read) in zip(self.r1, self.r2):
            worker = get_worker_info()
            worker_id = id(self) if worker is not None else -1
            yield (r1_read, r2_read), worker_id

    def __iter__(self):
        return zip(*[self.get_stream for _ in range(self.batch_size)])

    @classmethod
    def split_data(cls, r1, r2, batch_size, max_workers):
        # Determine the proper number of workers
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers
        return [cls(r1, r2, batch_size=split_size)]


class IterableSeqData(IterableDataset):
    '''
    @params:
        data: read generator
    '''

    def __init__(self, data, batch_size):
        # super(IterablePairedReadData).__init__()
        self.data = data
        self.batch_size = batch_size

    def get_stream(self):
        for seq in self.data:
            worker = get_worker_info()
            worker_id = id(self) if worker is not None else -1
            yield seq, worker_id

    # def get_batch(self):
    #     return zip(*[self.get_stream for _ in range(self.batch_size)])

    def __iter__(self):
        return zip(*[self.get_stream for _ in range(self.batch_size)])

    @classmethod
    def split_data(cls, data, batch_size, max_workers):
        # Determine the proper number of workers
        for n in range(max_workers, 0, -1):
            if batch_size % n == 0:
                num_workers = n
                break
        split_size = batch_size // num_workers
        return [cls(data, batch_size=split_size)]
