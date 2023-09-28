"""
Credits: https://github.com/felipecode/coiltraine/blob/master/configs/nocrash/resnet34imnet10S1.yaml
"""


SENSORS = {'rgb': (3, 88, 200)}
TARGETS = ['steer', 'throttle', 'brake']
INPUTS = ['speed_module']
NUMBER_FRAMES_FUSION = 1
PRE_TRAINED = True
MODEL_CONFIGURATION = {  # Based on the MODEL_TYPE, we specify the structure
    'perception': {  # The module that process the image input, it ouput the number of classes
        'res': {
            'name': 'resnet34',
            'num_classes': 512
        }
    },
    'measurements': { # The module the process the input float data, in this case speed_input
        'fc': { # Easy to configure fully connected layer
            'neurons': [128, 128], # Each position add a new layer with the specified number of neurons
            'dropouts': [0.0, 0.0]
        }
    },
    'join': { # The module that joins both the measurements and the perception
        'fc': {
            'neurons': [512],
            'dropouts': [0.0]
        }
    },
    'speed_branch': { # The prediction branch speed branch
        'fc': {
            'neurons': [256, 256],
            'dropouts': [0.0, 0.5]
        }
    },
    'branches': { # The output branches for the different possible directions ( Straight, Left, Right, None)
        'number_of_branches': 4,
        'fc': {
            'neurons': [256, 256],
            'dropouts': [0.0, 0.5]
        }
    }
}

