# -*- coding:utf-8 -*-
from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class CNNTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(CNNTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        losses = []
        accs = []
        curr_epoch = self.model.global_step_tensor.eval(self.sess)
        for epoch in range(curr_epoch, self.config.max_epoch):
            for batch in self.data.next_batch(self.config.batch_size, shuffle=True):
                results = self.train_step(batch)
                losses.append(results['loss'])
                accs.append(results['accuracy'])
            loss = np.mean(losses)
            acc = np.mean(accs)
            print(loss, ' ', acc)
            summaries_dict = {}
            summaries_dict['loss'] = loss
            summaries_dict['acc'] = acc
            self.logger.summarize(curr_epoch, summaries_dict=summaries_dict)

    def train_step(self, batch):
        feed_dict = {
            self.model.input_x: batch.sent,
            self.model.input_x_len: batch.sent_len,
            self.model.input_y: batch.label,
            self.model.drop_keep_rate: self.config.drop_keep_rate,
            self.model.learning_rate: self.config.learning_rate
        }
        to_return = {
            'train_step': self.model.train_step,
            'loss': self.model.loss,
            'accuracy': self.model.accuracy,
        }
        return self.sess.run(to_return, feed_dict)
