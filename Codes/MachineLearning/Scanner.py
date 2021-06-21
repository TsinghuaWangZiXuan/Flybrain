import multiprocessing

import numpy as np


class Scanner(object):
    """
    Class scanner is used to scan raw DNA sequence and find possible sites for tf to bind
    """

    def __init__(self, pwm, seq):
        self.pwm = pwm
        self.seq = seq
        self.tf_num = np.shape(self.pwm)[-1]
        self.tf_len = np.shape(self.pwm)[1]

    def get_tf_max_log_score(self):
        """
        Get max log score for each tf.
        :return: Max score vector. Shape=[2*tf_num]
        """
        score = np.zeros(self.tf_len)
        scores = []
        for i in range(self.tf_num):
            for j in range(self.tf_len):
                curr_tf = np.max(np.log(self.pwm[:, j, 0, i] + 1e-6))
                score[j] = curr_tf
            scores.append(np.sum(score))
        return scores

    def segment(self, seq, pwm, max_score):
        """
        Segmentation of DNA sequence for multiplying.
        :return: DNA sequence.
        """
        seg = np.swapaxes(np.asarray([seq[:, :, i:i + self.tf_len]
                                      for i in range(seq.shape[-1] - self.tf_len + 1)]), axis1=0, axis2=1)
        n = seg.shape[0]
        s = seg.shape[1]
        seg = np.reshape(seg, (n, s, -1))  # seg.shape = [n, s, 4*len]
        x = np.dot(seg, pwm)

        # x = np.squeeze(x, axis=1)  # tensor(N,S,2t)
        n = x.shape[0]
        s = x.shape[1]

        x = np.divide(np.exp(x), max_score)
        x = np.reshape(x, (n, s, 2, -1))

        m = x.shape[-1]
        x = np.reshape(x, (n, 2 * s, m))

        x = np.max(x, axis=1)
        return x

    def scan(self, seq, max_score, pwm, batch_size=32):
        """
        Scan DNA.
        :return: Vector for training.
        """
        batch_num = len(seq) // batch_size
        print(batch_num)
        x = []
        for indx in range(batch_num):
            s = seq[indx * batch_size: (indx + 1) * batch_size] if indx < batch_num - 1 else seq[
                                                                                             indx * batch_size:]
            x.append(self.segment(s, pwm, max_score))
            print(indx)
        x = np.concatenate(x, axis=0)
        return x

    def multi_scan(self, cpu_num=8):
        max_score = np.exp(self.get_tf_max_log_score())
        pwm = np.log(
            np.reshape(self.pwm, (self.pwm.shape[0] * self.pwm.shape[1], -1)) + 1e-6)  # pwm.shape = [4*len, 2*num]
        total_size = len(self.seq)
        data_size = total_size // cpu_num

        processes = []
        with multiprocessing.Pool(processes=cpu_num) as pool:
            for i in range(cpu_num):
                seq = self.seq[i * data_size:(i + 1) * data_size] if i < cpu_num - 1 else self.seq[
                                                                                          i * data_size:]
                p = pool.apply_async(self.scan, (seq, max_score, pwm))
                processes.append(p)
            pool.close()
            pool.join()

        x = []
        for p in processes:
            x.append(p.get())
        x = np.concatenate(x, axis=0)
        return x
