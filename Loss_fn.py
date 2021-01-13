import numpy as np
#import tensorflow as tf


class Loss:
    # def prob(self, a, t=2):
    #   return np.exp(a / t) / np.sum(np.exp(a / t))

    def l_dist(self, a, b, t=2):
        return - np.sum(np.exp(a / t) / np.sum(np.exp(a / t)) + np.log(np.exp(b / t) / np.sum(np.exp(b / t))))

    def l2_loss(self, vf, vs):
        vf /= np.linalg.norm(vf)
        vs /= np.linalg.norm(vs)
        return np.square(np.linalg.norm(vf - vs, ord=2))

    def loss_total(self, fd_vf, fd_vs, vf, vs, fg_vf, fg_vs, lambda_1=0.025, lambda_2=200):
        try:
            l1 = np.linalg.norm(fd_vf - fd_vs)
        except:
            l1 = 0
        l2 = lambda_1 * self.l2_loss(vf, vs)
        l3 = lambda_2 * self.l_dist(fg_vf, fg_vs)
        l_total = l1 + l2 + l3
        return l_total
