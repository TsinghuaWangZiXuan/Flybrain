import numpy as np
import keras.backend as K


class Scanner(object):
    def __init__(self, pwm, seq):
        self.pwm = pwm
        self.seq = seq
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
        print(0)
        seg = np.swapaxes(np.asarray([self.seq[:, :, i:i + self.tf_len]
                                      for i in range(self.seq.shape[-1] - self.tf_len + 1)]), axis1=0, axis2=1)
        print(1)
        n = seg.shape[0]
        s = seg.shape[1]
        seg = np.reshape(seg, (n, s, -1))  # seg.shape = [n, s, 4*len]
        pwm = np.reshape(self.pwm, (self.pwm.shape[0] * self.pwm.shape[1], -1))  # pwm.shape = [4*len, 2*num]
        return np.dot(seg, np.log(pwm+1e-6))

    def scan(self):
        max_score = self.get_tf_max_log_score()
        x = self.segment()  # (N,S,2t)
        print(2)
        n = x.shape[0]
        s = x.shape[1]

        x = np.divide(np.exp(x), np.exp(max_score))
        # x = np.exp(x)

        x = np.reshape(x, (n, s, 2, -1))
        return K.constant(x)
