import threading

class Optimizer(threading.Thread):

    def __init__(self, brain):
        threading.Thread.__init__(self)
        self.brain = brain
        self.stop_signal = False

    def run(self):
        while not self.stop_signal:
            self.brain.optimize()

    def stop(self):
        self.stop_signal = True