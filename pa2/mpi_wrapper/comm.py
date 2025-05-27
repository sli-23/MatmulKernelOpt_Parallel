from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        root = 0  
        if rank == root:
            np.copyto(dest_array, src_array)
            
            temp_array = np.empty_like(src_array)
            
            for i in range(1, size):
                self.comm.Recv(temp_array, source=i)
                
                if op == MPI.SUM:
                    dest_array += temp_array
                elif op == MPI.PROD:
                    dest_array *= temp_array
                elif op == MPI.MAX:
                    dest_array = np.maximum(dest_array, temp_array)
                elif op == MPI.MIN:
                    dest_array = np.minimum(dest_array, temp_array)
                
                data_size = temp_array.itemsize * temp_array.size
                self.total_bytes_transferred += data_size
            
            for i in range(1, size):
                self.comm.Send(dest_array, dest=i)
                
                data_size = dest_array.itemsize * dest_array.size
                self.total_bytes_transferred += data_size
        else:
            self.comm.Send(src_array, dest=root)
            
            data_size = src_array.itemsize * src_array.size
            self.total_bytes_transferred += data_size
            
            self.comm.Recv(dest_array, source=root)
            
            data_size = dest_array.itemsize * dest_array.size
            self.total_bytes_transferred += data_size

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via non-blocking communication.
            
        The total data transferred is updated for each pairwise exchange.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        
        assert src_array.size % size == 0, "src_array size must be divisible by the number of processes"
        assert dest_array.size % size == 0, "dest_array size must be divisible by the number of processes"

        segment_size = src_array.size // size
        segment_bytes = segment_size * src_array.itemsize
        
        np.copyto(dest_array[rank*segment_size:(rank+1)*segment_size], 
                 src_array[rank*segment_size:(rank+1)*segment_size])
        
        num_requests = 2 * (size - 1) 
        requests = [None] * num_requests
        req_idx = 0
        
        segments = [(i*segment_size, (i+1)*segment_size) for i in range(size)]
        
        for i in range(size):
            if i != rank:
                start, end = segments[i]
                recv_segment = dest_array[start:end]
                requests[req_idx] = self.comm.Irecv(recv_segment, source=i)
                req_idx += 1
        
        for i in range(size):
            if i != rank:
                start, end = segments[i]
                send_segment = src_array[start:end]
                requests[req_idx] = self.comm.Isend(send_segment, dest=i)
                req_idx += 1
        self.total_bytes_transferred += 2 * (size - 1) * segment_bytes
        
        if requests:
            MPI.Request.Waitall(requests)