import numpy as np
import keras.backend as K


class Scanner(object):
    """
    Class scanner is used to scan raw DNA sequence and find possible sites for tf to bind
    """
    def __init__(self, PWM, Seq):
        self.pwm = PWM
        self.seq = np.squeeze(Seq)
        self.tf_num = np.shape(self.pwm)[-1]
        self.tf_len = np.shape(self.pwm)[1]

    def get_tf_max_log_score(self):
        score = np.zeros(self.tf_len)
        scores = []
        for i in range(self.tf_num):
            for j in range(self.tf_len):
                curr_tf = np.max(np.log(self.pwm[:, j, 0, i] + 1e-6))
                score[j] = curr_tf
            scores.append(np.sum(score))
        return scores

    def segment(self):
        """
        Segmentation of DNA sequence for multiplying.
        :return: DNA sequence.
        """
        seg = np.swapaxes(np.asarray([self.seq[:, :, i:i + self.tf_len]
                                      for i in range(self.seq.shape[-1] - self.tf_len + 1)]), axis1=0, axis2=1)
        n = seg.shape[0]
        s = seg.shape[1]
        seg = np.reshape(seg, (n, s, -1))  # seg.shape = [n, s, 4*len]
        pwm = np.squeeze(self.pwm)
        pwm = np.reshape(pwm, [-1, self.tf_num])
        x = np.dot(seg, np.log(pwm + 1e-6))

        return x

    def scan(self):
        tf_max_log_score = self.get_tf_max_log_score()
        x = self.segment()
        N = x.shape[0]
        S = x.shape[1]

        x = np.divide(np.exp(x), np.exp(tf_max_log_score))
        x = np.reshape(x, (N, S, 2, -1))
        return K.constant(x)
