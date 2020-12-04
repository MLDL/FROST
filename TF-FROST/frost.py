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
import functools
import os
import json
from collections import defaultdict
import time
import string
import random

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

from cta_frost.cta_remixmatch import CTAReMixMatch
from libml import data, utils, augment, ctaugment
import objective as obj_lib
from collections import defaultdict
from third_party import data_util

FLAGS = flags.FLAGS


class AugmentPoolCTACutOut(augment.AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=x)
        assert not probe
        cutout_policy = lambda: cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cutout_policy()) for y in x[1:]]).astype('f'))


class Frost(CTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    def train(self, train_nimg, report_nimg, numPerClass, wclr):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return
        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(batch * self.params['uratio']).prefetch(16)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))
        name = ''
        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()
        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                for _ in loop:
                    self.train_step(train_session, gen_labeled, gen_unlabeled, wclr)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))
            if numPerClass > 1:
                self.call_bootstrap(numPerClass=numPerClass)

        del train_labeled, train_unlabeled, scaffold, train_session
        return

####################### Modification 
    def class_balancing(self, pseudo_labels, balance, confidence, delT):

        if balance > 0:
            pLabels = tf.math.argmax(pseudo_labels, axis=1)
            pLabels = tf.cast(pLabels,dtype=tf.float32)
            classes, idx, counts = tf.unique_with_counts(pLabels)
            shape = tf.constant([self.dataset.nclass])
            classes = tf.cast(classes,dtype=tf.int32)
            class_count = tf.scatter_nd(tf.reshape(classes,[tf.size(classes),1]),counts, shape)

            class_count = tf.cast(class_count,dtype=tf.float32)
            mxCount = tf.reduce_max(class_count, axis=0)

            pLabels = tf.cast(pLabels,dtype=tf.int32)
            if balance == 1 or balance == 4:
                confidences = tf.zeros_like(pLabels,dtype=tf.float32)
                ratios  = 1.0 - tf.math.divide_no_nan(class_count, mxCount)
                ratios  = confidence - delT*ratios
                confidences = tf.gather_nd(ratios, tf.reshape(pLabels,[tf.size(pLabels),1]) )
                pseudo_mask = tf.reduce_max(pseudo_labels, axis=1) >= confidences
            else:
                pseudo_mask = tf.reduce_max(pseudo_labels, axis=1) >= confidence

            if balance == 3 or balance == 4:
                classes, idx, counts = tf.unique_with_counts(tf.boolean_mask(pLabels,pseudo_mask))
                shape = tf.constant([self.dataset.nclass])
                classes = tf.cast(classes,dtype=tf.int32)
                class_count = tf.scatter_nd(tf.reshape(classes,[tf.size(classes),1]),counts, shape)
                class_count = tf.cast(class_count,dtype=tf.float32)
            pseudo_mask = tf.cast(pseudo_mask,dtype=tf.float32)

            if balance > 1:
                ratios  = tf.math.divide_no_nan(tf.ones_like(class_count,dtype=tf.float32),class_count)
                ratio = tf.gather_nd(ratios, tf.reshape(pLabels,[tf.size(pLabels),1]) )
                Z = tf.reduce_sum(pseudo_mask)
                pseudo_mask = tf.math.multiply(pseudo_mask, tf.cast(ratio,dtype=tf.float32))
                pseudo_mask = tf.math.divide_no_nan(tf.scalar_mul(Z, pseudo_mask), tf.reduce_sum(pseudo_mask))
        else:
            pseudo_mask = tf.cast(tf.reduce_max(pseudo_labels, axis=1) >= confidence,dtype=tf.float32)

        return pseudo_mask 

###################### End
           

    def model(self, batch, lr, wd, wu, wclr,  mom, confidence, balance, delT, uratio, clrratio, temperature, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')  # Training unlabeled (weak, strong)
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels
        wclr_in = tf.placeholder(tf.int32, [1], 'wclr')  # wclr

        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = utils.interleave(tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        logits = utils.de_interleave(logits, 2 * uratio+1)
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:], 2)
        del logits, skip_ops

        # Labeled cross-entropy
        loss_xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=l_in, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        tf.summary.scalar('losses/xe', loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
        loss_xeu = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(pseudo_labels, axis=1),
                                                                  logits=logits_strong)
#        pseudo_mask = tf.to_float(tf.reduce_max(pseudo_labels, axis=1) >= confidence)
        pseudo_mask = self.class_balancing(pseudo_labels, balance, confidence, delT)
        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        loss_xeu = tf.reduce_mean(loss_xeu * pseudo_mask)
        tf.summary.scalar('losses/xeu', loss_xeu)

####################### Modification
        # Contrastive loss term
        contrast_loss = 0
        if wclr > 0 and wclr_in == 0:
            ratio = min(uratio, clrratio)
            if FLAGS.clrDataAug == 1:
                preprocess_fn = functools.partial(data_util.preprocess_for_train, height=self.dataset.height, width=self.dataset.width)
                x = tf.concat([lambda y: preprocess_fn(y), lambda y: preprocess_fn(y)], 0)
                embeds = lambda x, **kw: self.classifier(x, **kw, **kwargs).embeds
                hidden = utils.para_cat(lambda x: embeds(x, training=True), x)
            else:
                embeds = lambda x, **kw: self.classifier(x, **kw, **kwargs).embeds
                hiddens = utils.para_cat(lambda x: embeds(x, training=True), x)
                hiddens = utils.de_interleave(hiddens, 2 * uratio + 1)
                hiddens_weak, hiddens_strong = tf.split(hiddens[batch:], 2, 0)
                hidden = tf.concat([hiddens_weak[:batch*ratio], hiddens_strong[:batch*ratio]], axis=0)
                del hiddens, hiddens_weak, hiddens_strong

            contrast_loss, _, _  = obj_lib.add_contrastive_loss(
                hidden, hidden_norm=True, # FLAGS.hidden_norm,
                temperature=temperature, tpu_context=None)

            tf.summary.scalar('losses/contrast', contrast_loss)
            del embeds, hidden
###################### End

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

#        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
        train_op = tf.train.MomentumOptimizer(lr, mom, use_nesterov=True).minimize(
            loss_xe + wu * loss_xeu + wclr*contrast_loss + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, wclr=wclr_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

####################### New code
class CreateSplit():
    def __init__(self, train_dir: str,  **kwargs):
        self.train_dir = os.path.join(train_dir) # , self.experiment_name(**kwargs))
        self.params = utils.EasyDict(kwargs)
        self.session = None
        self.tmp = utils.EasyDict(print_queue=[], cache=utils.EasyDict())


    def get_class(self, serialized_example):
        return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def create_split(self, datasetName, seed, size):
        input_file=data.DATA_DIR+'/'+datasetName+'-train.tfrecord'
        target = ('%s/%s/%s.%d@%d' % (data.DATA_DIR, FLAGS.data_subfolder, datasetName, seed, size) )
        print("target ",target)

        count = 0
        id_class = []
        class_id = defaultdict(list)
        dataset = tf.data.TFRecordDataset(input_file).map(self.get_class, 4).batch(1 << 10)
        it = dataset.make_one_shot_iterator().get_next()
        try:
#            with tf.Session() as session, tqdm(leave=False) as t:
            with tf.Session() as session:
                while 1:
                    for i in session.run(it):
                        id_class.append(i)
                        class_id[i].append(count)
                        count += 1
        except tf.errors.OutOfRangeError:
            pass
#        print('%d records found' % count)
        nclass = len(class_id)
        assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
        train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64)
        train_stats /= train_stats.max()
        if 'stl10' in self.train_dir:
            # All of the unlabeled data is given label 0, but we know that
            # STL has equally distributed data among the 10 classes.
            train_stats[:] = 1

#        print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
        assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
        class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)]

        # Distribute labels to match the input distribution.
        train_data = np.load(target + ".npy")
        samples = []
        pseudo_labels = []
        for i in range(size):
            c = i%nclass
            npos = i//nclass
            pseudo_labels.append(c)
            samples.append(train_data[c][npos])
#            print("Label ",id_class[train_data[c][npos]]," Pseudo-label ", c, " Top confidence ", npos, " train_data ", train_data[c][npos])

        del train_data
        label = frozenset([int(x) for x in samples])

        if 'stl10' in self.train_dir and size == 1000:
            train_data = tf.gfile.Open(os.path.join(data.DATA_DIR, 'stl10_fold_indices.txt'), 'r').read()
            label = frozenset(list(map(int, data.split('\n')[seed].split())))

        print('Creating split in %s' % target)
        tf.gfile.MakeDirs(os.path.dirname(target))
        with tf.python_io.TFRecordWriter(target + '-label.tfrecord') as writer_label:
            pos, loop = 0, trange(count, desc='Writing records')
            for record in tf.python_io.tf_record_iterator(input_file):
                if pos in label:
                    pseudo_label = pseudo_labels[samples.index(pos)]
                    feat = dict(image=self._bytes_feature(tf.train.Example.FromString(record).features.feature['image'].bytes_list.value[0]),
                                label=self._int64_feature(pseudo_label))
                    newrecord = tf.train.Example(features=tf.train.Features(feature=feat))
                    writer_label.write(newrecord.SerializeToString())
                pos += 1
                loop.update()
            loop.close()
        with tf.gfile.Open(target + '-label.json', 'w') as writer:
            writer.write(json.dumps(dict(distribution=train_stats.tolist(), label=sorted(label)), indent=2, sort_keys=True))

###################### End

def main(argv):
    utils.setup_main()
    del argv  # Unused.
    seedIndx = FLAGS.dataset.find('@')
    seed = int(FLAGS.dataset[seedIndx-1])

    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = Frost(
        os.path.join(FLAGS.train_dir, dataset.name, Frost.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        wclr=FLAGS.wclr,
        mom=FLAGS.mom,
        confidence=FLAGS.confidence,
        balance=FLAGS.balance,
        delT=FLAGS.delT,
        uratio=FLAGS.uratio,
        clrratio=FLAGS.clrratio,
        temperature=FLAGS.temperature,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
###################### New code
    tic = time.perf_counter()
    if FLAGS.boot_factor > 1:
        numIter = 2
        numToLabel = [FLAGS.boot_factor, FLAGS.boot_factor*FLAGS.boot_factor, 0]
        numImgs = [(FLAGS.train_kimg << 9), 3*(FLAGS.train_kimg << 8), (FLAGS.train_kimg << 10) ]
        if FLAGS.boot_schedule == 1:
            steps = int((FLAGS.train_kimg << 10) / 3)
            numImgs = [steps, 2*steps, 3*steps]
        elif FLAGS.boot_schedule == 2:
            numIter = 3
            steps = FLAGS.train_kimg << 8
            numImgs = [steps, 2*steps, 3*steps, 4*steps]
            numToLabel = [FLAGS.boot_factor, FLAGS.boot_factor**2,FLAGS.boot_factor**3, 0]

        datasetName = dataset.name[:dataset.name.find('.')]
        print("Dataset Name ", datasetName)
        letters = string.ascii_letters
        subfolder = ''.join(random.choice(letters) for i in range(8))
        FLAGS.data_subfolder = subfolder
        tf.gfile.MakeDirs(data.DATA_DIR+'/'+subfolder)
        if not tf.gfile.Exists(data.DATA_DIR+'/'+subfolder+'/'+datasetName+'-unlabel.json'):
            infile = data.DATA_DIR+'/SSL2/'+datasetName+'-unlabel.'
            outfile = data.DATA_DIR+'/'+subfolder+'/'+datasetName+'-unlabel.'
            print("Copied from ",infile, "* to ", outfile +'*')
            tf.io.gfile.copy(infile+'json', outfile + 'json')
            tf.io.gfile.copy(infile+'tfrecord', outfile + 'tfrecord')

        for it in range(numIter):
            print(" Iiteration ", it, " until ", numImgs[it])
            model.train(numImgs[it], FLAGS.report_kimg << 10, numToLabel[it], it)
            elapse = (time.perf_counter() - tic) / 3600
            print("After iteration ", it, " training time ", elapse, " hours")

            bootstrap = CreateSplit(
                os.path.join(FLAGS.train_dir, dataset.name, Frost.cta_name()))
            bootstrap.create_split(datasetName=datasetName, seed=seed, size=numToLabel[it]*dataset.nclass)

            target = datasetName + '.' + str(seed) + '@'  + str(numToLabel[it]*dataset.nclass) + '-1'
            print("Target ", target)
            dataset = data.PAIR_DATASETS()[target]()
            log_width = utils.ilog2(dataset.width)
            model.updateDataset(dataset)

        print(" Iiteration 2 until ", numImgs[numIter])
        model.train(numImgs[numIter], FLAGS.report_kimg << 10, 0, numIter)
        tf.compat.v1.gfile.DeleteRecursively(data.DATA_DIR+'/'+subfolder)
    else:
        model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10, 0, 0)

    elapse = (time.perf_counter() - tic) / 3600
    print(f"Total training time {elapse:0.4f} hours")
###################### End

if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_float('wclr', 0, 'contrastive loss weight.')
    flags.DEFINE_float('mom', 0.9, 'Momentum coefficient.')
    flags.DEFINE_float('temperature', 1,'Temperature parameter for contrastive loss.')
    flags.DEFINE_float('delT', 0.2, 'The amount balance=1 can reduce the confidence threshold.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    flags.DEFINE_integer('clrratio', 1, 'Unlabeled batch size ratio.')
    flags.DEFINE_integer('boot_factor', 8,'Factor for increasing the number of labeled data')
    flags.DEFINE_integer('boot_schedule', 0,'Schedule for increasing the number of labeled data')
    flags.DEFINE_integer('balance', 0, 'Method to help balance classes')
    flags.DEFINE_integer('clrDataAug', 0, 'Method for CLR data augmentation')
    flags.DEFINE_boolean('hidden_norm', True,'Temperature parameter for contrastive loss.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    app.run(main)
