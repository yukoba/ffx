#!/usr/bin/env python

import numpy as np
import ffx
import math

# This creates a dataset of 1 predictor
train_X = np.array([np.arange(0, 6, 0.01)]).T
train_y = np.array([math.sin(x) for x in np.arange(0, 6, 0.01)])

test_X = np.array([np.arange(0, 6, 0.011)]).T
test_y = np.array([math.sin(x) for x in np.arange(0, 6, 0.011)])

models = ffx.run(train_X, train_y, test_X, test_y, ["x"], False)

print('True model: y = sin(x)')
print('Results:')
print('Num bases,Test error (%),Model\n')
for model in models:
    print('%10s, %13s, %s\n' %
          ('%d' % model.numBases(), '%.4f' % (model.test_nmse * 100.0), model))
