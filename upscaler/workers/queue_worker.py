from multiprocessing import Queue
from threading import Event, Thread


class QueueWorker(Thread):
    def __init__(self, queue=None):
        Thread.__init__(self)

        self.daemon = True
        self.exit = Event()

        if not queue:
            queue = Queue()
        self.queue = queue

    def run(self) -> None:
        while not self.exit.is_set():
            try:
                data = self.queue.get()
                self.process_work(data)
            except Exception as e:
                print(e)
                raise
                continue

    def process_work(self, encoding_job):
        pass
