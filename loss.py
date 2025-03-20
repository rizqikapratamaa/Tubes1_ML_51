import math
from Node import Node

# cross categorical entropy
def cce(outputs, targets, output_size):
    loss = Node(0.0)
    batch_size = len(outputs)
    for i in range(batch_size):
        loss = loss + (Node(-targets[i]) * Node(math.log(outputs[i].value + 1e-15)))
    loss = loss * Node(1.0 / output_size)
    return loss

# binary cross entropy
def bce(outputs, targets):
    loss = Node(0.0)
    batch_size = len(targets) 

    for i in range(batch_size):
        prob = outputs[i].value 
        target = targets[i]
        
        # clipping
        prob = max(min(prob, 1 - 1e-15), 1e-15)

        # bce formula: - (y log(p) + (1 - y) log(1 - p))
        loss = loss + Node(-target) * Node(math.log(prob)) + Node(-(1 - target)) * Node(math.log(1 - prob))

    loss = loss * Node(1.0 / batch_size) 
    return loss