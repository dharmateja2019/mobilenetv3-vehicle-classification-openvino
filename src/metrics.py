import time

class Meter:
    def __init__(self):
        self.start = None
        self.end = None

    def tic(self):
        self.start = time.time()

    def toc(self):
        self.end = time.time()

    def latency_ms(self):
        return (self.end - self.start) * 1000

    def fps(self):
        return 1 / (self.end - self.start)
