import tensorflow as tf
import keras.backend as K
import numpy as np


class Scanner(tf.keras.initializers.Initializer):
    """
    Class scanner is used to scan raw DNA sequence and find possible sites for tf to bind
    """

    def __init__(self, pwm):
        self.pwm = pwm
        self.tf_num = np.shape(self.pwm)[-1]
        self.tf_len = np.shape(self.pwm)[1]
        self.motif = self.pwm
        for i in range(self.tf_num):
            for j in range(self.tf_len):
                # Calculate maximum
                curr_tf = np.max(self.pwm[:, j, 0, i])
                for b in range(4):
                    # Scale
                    self.motif[b, j, 0, i] /= curr_tf

        self.motif = np.log(self.motif)
        self.motif[np.isinf(self.motif)] = -1e-6

    def __call__(self, shape, dtype=None):
        """
        Motif for segmentation layer.
        :return: Motif.
        """
        return K.constant(self.motif, dtype=dtype)
