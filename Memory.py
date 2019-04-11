from collections import deque  # Ordered collection with ends
import numpy as np

class Memory():


    # Init deque for the memory:
    def __init__(self,max_size):
        self.buffer = deque(maxlen= max_size)

    # Add experience to memory:
    def add(self, experience):
        self.buffer.append(experience)

    # Take random batch_size experiences from memory:
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),size=batch_size,replace=False)

        return [self.buffer[i] for i in index]

    # Get all experiences:
    def getAllMemory(self):
        return self.buffer

    # Get the size of the memory:
    def getMemorySize(self):
        return len(self.buffer)

    # Get max size of the memory:
    def getCapacity(self):
        return self.buffer.maxlen


