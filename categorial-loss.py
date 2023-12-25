import math
import numpy as np

softmax_output = [0.7, 0.1, 0.2] # y hat
target_output = [1,0,0] # y

loss = -(math.log(softmax_output[0])*target_output[0] +
math.log(softmax_output[1])*target_output[1] +
math.log(softmax_output[2])*target_output[2])

loss = -math.log(softmax_output[0])

#print(loss) # 0.35667494393873245

# Probabilities for 3 samples
softmax_outputs = np.array([
    [0.7, 0.1, 0.2], # dog
    [0.1, 0.5, 0.4], # cat
    [0.02, 0.9, 0.08]]) # cat
class_targets = [0, 1, 1] # dog, cat, cat

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
#print(neg_log) # [0.35667494 0.69314718 0.10536052]
average_loss = np.mean(neg_log)
#print(average_loss) # 0.38506088005216804

softmax_outputs = np.array([[0.7, 0.1, 0.2],
[0.1, 0.5, 0.4],
[0.02, 0.9, 0.08]])
class_targets = np.array([[1, 0, 0],
[0, 1, 0],
[0, 1, 0]])

# Probabilities for target values -
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
    range(len(softmax_outputs)),
    class_targets
    ]
# Mask values - only for one-hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
    softmax_outputs * class_targets,
    axis=1)
    
# Losses
neg_log = -np.log(correct_confidences)
average_loss = np.mean(neg_log)
print(average_loss)