import csv
from threading import Thread
from queue import Queue, Empty


class IterationStats:

    def __init__(self, filename, values_type=float, value_type=float, dims=1):
        self.filename = filename
        self.iterations = []
        self.write_thread = None
        self.values_type = values_type
        self.value_type = value_type
        self.dims = dims
    
    def start_writing(self):
        open(self.filename, 'w').close()
        self.write_thread = IterationStats.WriteThread(self.filename, self.dims)
        self.write_thread.start()

    def save_iteration(self, iteration_number, iteration_time, iteration_value, values):
        self.write_thread.write(iteration_number, iteration_time, iteration_value, values)

    def done_writing(self):
        self.write_thread.done()
        self.write_thread.join()

    def load_stats(self):
        with open(self.filename, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            while True:
                iteration_info = next(reader, None)
                if iteration_info is None:
                    break
                iteration_number = int(iteration_info[0])
                iteration_time = float(iteration_info[1])
                iteration_value = float(iteration_info[2])
                if self.dims == 1:
                    values_str = next(reader)
                    values = [float(v) for v in values_str]
                else:
                    values = []
                    for _ in range(0, self.dims):
                        values_str = next(reader)
                        values.append([float(v) for v in values_str])
                self.iterations.append({'number': iteration_number, 'time': iteration_time, 'i_value': iteration_value, 'values': values})

    def run_analysis(self, on_iteration):
        for iteration_info in self.iterations:
            number = iteration_info['number']
            time = iteration_info['time']
            i_value = iteration_info['i_value']
            values = iteration_info['values']
            on_iteration(number, time, i_value, values)
    
    def load_and_run_analysis(self, on_iteration):
        with open(self.filename, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            while True:
                iteration_info = next(reader, None)
                if iteration_info is None:
                    break
                iteration_number = int(iteration_info[0])
                iteration_time = float(iteration_info[1])
                iteration_value = self.value_type(iteration_info[2])
                if self.dims == 1:
                    values_str = next(reader)
                    values = [float(v) for v in values_str]
                else:
                    values = []
                    for _ in range(0, self.dims):
                        values_str = next(reader)
                        values.append([float(v) for v in values_str])
                on_iteration(iteration_number, iteration_time, iteration_value, values)

    class WriteThread(Thread):

        def __init__(self, filename, dims):
            super().__init__()
            self.write_queue = Queue()
            self.filename = filename
            self.running = True
            self.dims = dims

        def done(self):
            self.running = False

        def write(self, iteration_number, iteration_time, iteration_value, values):
            self.write_queue.put((iteration_number, iteration_time, iteration_value, values))
        
        def run(self):
            while self.running or not self.write_queue.empty():
                try:
                    iteration_number, iteration_time, iteration_value, values = self.write_queue.get(timeout=1)
                except Empty:
                    continue

                with open(self.filename, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=' ')
                    writer.writerow([iteration_number, iteration_time, iteration_value])
                    if self.dims == 1:
                        writer.writerow(values)
                    else:
                        for d in range(0, self.dims):
                            writer.writerow(values[d])
