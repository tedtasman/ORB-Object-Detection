'''
Lion Sight Emulator
Author: Ted Tasman
Date: 2025-03-26

This module emulates the functionality of lion_sight.py for testing purposes.
'''

class LionSight:

    def __init__(self, num_targets=4, corners=None, predetermined_coordinate=None):

        self.num_targets = num_targets
        self.boundaries = corners
        self.predetermined_coordinate = predetermined_coordinate
        self.targets = None
    

    def detect_targets(self):
        '''
        Returns the coordinates of the targets
        '''

        if self.predetermined_coordinate:
            return self.predetermined_coordinate
        else:
            return self.targets

