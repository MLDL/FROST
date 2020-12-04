# Copyright 2019 Google LLC
# Modified 2020, FROST
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf
import numpy as np
from absl import flags
import statistics 
import os
import string
import random

from fully_supervised.lib.train import ClassifyFullySupervised
from libml import data
from libml.augment import AugmentPoolCTA
from libml.ctaugment import CTAugment
from libml.train import ClassifySemi
from tqdm import trange, tqdm

FLAGS = flags.FLAGS

flags.DEFINE_integer('adepth', 2, 'Augmentation depth.')
flags.DEFINE_float('adecay', 0.99, 'Augmentation decay.')
flags.DEFINE_float('ath', 0.80, 'Augmentation threshold.')


class CTAClassifySemi(ClassifySemi):
    """Semi-supervised classification."""
    AUGMENTER_CLASS = CTAugment
    AUGMENT_POOL_CLASS = AugmentPoolCTA

    @classmethod
    def cta_name(cls):
        return '%s_depth%d_th%.2f_decay%.3f' % (cls.AUGMENTER_CLASS.__name__,
                                                FLAGS.adepth, FLAGS.ath, FLAGS.adecay)

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int, **kwargs):
        ClassifySemi.__init__(self, train_dir, dataset, nclass, **kwargs)
        self.augmenter = self.AUGMENTER_CLASS(FLAGS.adepth, FLAGS.ath, FLAGS.adecay)
        self.best_acc=0
        self.best_accStd=0
        self.counter=0

    def updateKeywords(self, **kwargs):
        print("New arguements")
        for k, v in sorted(kwargs.items()):
            self.kwargs[k] = v
            print('%-32s %s' % (k, v))
        print("updated arguements")
        for k, v in sorted(self.kwargs.items()):
            print('%-32s %s' % (k, v))

    def updateDataset(self, dataset):
        self.dataset = dataset
        print("New dataset name ", dataset.name)

    def gen_labeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = True
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def gen_unlabeled_fn(self, data_iterator):
        def wrap():
            batch = self.session.run(data_iterator)
            batch['cta'] = self.augmenter
            batch['probe'] = False
            return batch

        return self.AUGMENT_POOL_CLASS(wrap)

    def train_step(self, train_session, gen_labeled, gen_unlabeled, wclr):
        x, y = gen_labeled(), gen_unlabeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.y: y['image'],
                                         self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.wclr: [wclr],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)

    def cache_eval(self):
        """Cache datasets for computing eval stats."""

        def collect_samples(dataset, name):
            """Return numpy arrays of all the samples from a dataset."""
#            pbar = tqdm(desc='Caching %s examples' % name)
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            images, labels = [], []
            while 1:
                try:
                    v = self.session.run(it)
                except tf.errors.OutOfRangeError:
                    break
                images.append(v['image'])
                labels.append(v['label'])
#                pbar.update()

            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
#            pbar.close()
            return images, labels

        if 'test' not in self.tmp.cache:
            self.tmp.cache.test = collect_samples(self.dataset.test.parse(), name='test')
            self.tmp.cache.valid = collect_samples(self.dataset.valid.parse(), name='valid')
            self.tmp.cache.train_labeled = collect_samples(self.dataset.train_labeled.take(10000).parse(),
                                                           name='train_labeled')
            self.tmp.cache.train_original = collect_samples(self.dataset.train_original.parse(),
                                                           name='train_original')

    def eval_stats(self, batch=None, feed_extra=None, classify_op=None, verbose=True):
        """Evaluate model on train, valid and test."""
        batch = batch or FLAGS.batch
        classify_op = self.ops.classify_op if classify_op is None else classify_op
        accuracies = []
        class_acc = {}
        best_class_acc = {}
        for subset in ('train_labeled', 'valid', 'test'):
            images, labels = self.tmp.cache[subset]
            predicted = []
            for x in range(0, images.shape[0], batch):
                p = self.session.run(
                    classify_op,
                    feed_dict={
                        self.ops.x: images[x:x + batch],
                        **(feed_extra or {})
                    })
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            pred = predicted.argmax(1)
            probs = predicted.max(1)
            accuracies.append((pred == labels).mean() * 100)
#####  New Code 

        testAcc = float(accuracies[2])
        if testAcc  > self.best_acc:
            self.best_acc = testAcc

        if verbose:
            epochs = self.tmp.step // 5000
            acc = list([epochs/10] + [self.tmp.step >> 10] + accuracies)
            acc.append(self.best_acc)
            tup = tuple(acc)
            self.train_print('Epochs %d, kimg %-5d  accuracy train/valid/test/best_test  %.2f  %.2f  %.2f  %.2f  ' % tup)
##### End of new code

#        if verbose:
#            self.train_print('kimg %-5d  accuracy train/valid/test  %.2f  %.2f  %.2f' %
#                             tuple([self.tmp.step >> 10] + accuracies))
#        self.train_print(self.augmenter.stats())
        return np.array(accuracies, 'f')

    #####  New Code to compute class accuracies
    def get_random_string(self, length):
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))

        return result_str

    def bootstrap(self, classify_op=None, numPerClass=16):
        """Output the highest confidence pseudo-labeled examples."""

        classify_op = self.ops.classify_raw if classify_op is None else classify_op
        accuracies = []
        class_acc = {}
        best_class_acc = {}
        images, labels = self.tmp.cache['train_original']
        batch = FLAGS.batch # len(labels)//10
        predicted = []
        for x in range(0, images.shape[0], batch):
            p = self.session.run(
                classify_op,
                feed_dict={
                    self.ops.x: images[x:x + batch]
                })
            predicted.append(p)
        predicted = np.concatenate(predicted, axis=0)
        preds = predicted.argmax(1)
        probs = predicted.max(1)
        top = np.argsort(-probs,axis=0)

        unique_train_pseudo_labels, unique_train_counts = np.unique(preds, return_counts=True)
        print("Number of training pseudo-labels in each class: ", unique_train_counts," for classes: ", unique_train_pseudo_labels)

        sortByClass = np.random.randint(0,high=len(labels), size=(self.nclass, numPerClass), dtype=int)
        indx = np.zeros([self.nclass], dtype=int)
        matches = np.zeros([self.nclass, numPerClass], dtype=int)
        labls  = preds[top]
        samples = top
        for i in range(len(top)):
            if indx[labls[i]] < numPerClass:
                sortByClass[labls[i], indx[labls[i]]] = samples[i]
                if labls[i] == labels[top[i]]:
                    matches[labls[i], indx[labls[i]]] = 1
                indx[labls[i]] += 1

        if min(indx) < numPerClass:
            print("Counts of at least one class ", indx, " is lower than ", numPerClass)
        seedIndx = self.train_dir.find('@')
        seed = self.train_dir[seedIndx-1]
        size = numPerClass * self.nclass

        datasetName = self.dataset.name[:self.dataset.name.find('.')]
        target =  '%s.%s@%d.npy' % (datasetName,seed, size)
        target = '%s/%s/%s' % (data.DATA_DIR, FLAGS.data_subfolder, target)
        print("Saving ", target)
        np.save(target, sortByClass[0:self.nclass, :numPerClass])

        classAcc = 100*np.sum(matches, axis=1)/numPerClass
        print("Accuracy of the predicted pseudo-labels: top ", numPerClass,  ", ", np.mean(classAcc), classAcc )

##### End of new code
        return

class CTAClassifyFullySupervised(ClassifyFullySupervised, CTAClassifySemi):
    """Fully-supervised classification."""

    def train_step(self, train_session, gen_labeled):
        x = gen_labeled()
        v = train_session.run([self.ops.classify_op, self.ops.train_op, self.ops.update_step],
                              feed_dict={self.ops.x: x['probe'],
                                         self.ops.xt: x['image'],
                                         self.ops.label: x['label']})
        self.tmp.step = v[-1]
        lx = v[0]
        for p in range(lx.shape[0]):
            error = lx[p]
            error[x['label'][p]] -= 1
            error = np.abs(error).sum()
            self.augmenter.update_rates(x['policy'][p], 1 - 0.5 * error)
