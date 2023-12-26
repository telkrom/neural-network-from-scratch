import numpy as np

layer_outputs = [4.8, 1.21, 2.385]
exp_values = np.exp(layer_outputs)
#print("Exponentianed values:", )
#print(exp_values)

norm_values = exp_values / np.sum(exp_values)
#print("Normalized values:")
#print(norm_values)
#print(np.sum(norm_values))

# CATEGORICAL CROSS-ENTROPY
import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1,0,0]
loss = -math.log(softmax_output[0])
#print(loss)# 0.35667494393873245

a =  [0.22, 0.6, 0.18]
b =  [0.32, 0.36, 0.32]
l1 = -math.log(a[1])
l2 = -math.log(b[1])
#print(l1, l2)

# output of softmax activation function with a batch of 3 samples and 3 classes
softmax_outputs = np.array([
    [0.7, 0.1, 0.2],    # dog
    [0.1, 0.5, 0.4],    # cat
    [0.02, 0.9, 0.08]]) # cat
class_targets = [0, 1, 1]

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log)

softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]])
class_targets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]])

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
else:
    correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)

neg_log = -np.log(correct_confidences)
#print(neg_log)
average_loss = np.mean(neg_log)
#print(average_loss)


# Probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
[0.5, 0.1, 0.4],
[0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])
predictions = np.argmax(softmax_outputs, axis=1) 
acc = np.mean(predictions == class_targets)
print(acc)