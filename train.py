# -*- coding: utf-8 -*-
#

import os
import numpy as np
import argparse
import chainer
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

from data import Pix2pixDataset, Pix2pixIterator
from net import UNetGenerator, P2PDiscriminator

class Pix2pixUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(Pix2pixUpdater, self).__init__(*args, **kwargs)

    def loss_gen(self, gen, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize, _, w, h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, gen)
        return loss

    def loss_dis(self, dis, y_in, y_out):
        batchsize, _, w, h = y_in.data.shape

        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):
        gen_opt = self.get_optimizer('gen')
        dis_opt = self.get_optimizer('dis')
        gen, dis = self.gen, self.dis
        xp = gen.xp

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        in_ch = batch[0][0].shape[0]
        out_ch = batch[0][1].shape[0]
        w_in = 256
        w_out = 256

        x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype('f')
        t_out = xp.zeros((batchsize, out_ch, w_in, w_in)).astype('f')

        for i in range(batchsize):
            x_in[i, :] = xp.asarray(batch[i][0])
            t_out[i, :] = xp.asarray(batch[i][1])
        
        x_in = chainer.Variable(x_in)
        x_out = gen(x_in)

        y_fake = dis(x_in, x_out)
        y_real = dis(x_in, t_out)

        gen_opt.update(self.loss_gen, gen, x_out, t_out, y_fake)
        x_in.unchain_backward()
        x_out.unchain_backward()
        dis_opt.update(self.loss_dis, dis, y_real, y_fake)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    
    parser.add_argument("--epoch", '-e', type=int, default=200)
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    parser.add_argument("--display-interval", type=int, default=100)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    gen = UNetGenerator(args.ngf)
    dis = P2PDiscriminator(args.ndf)
    train_data = Pix2pixDataset(args.input_dir)
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    opt_gen = chainer.optimizers.Adam(alpha=args.lr, beta1=args.beta1)
    opt_gen.setup(gen)
    opt_dis = chainer.optimizers.Adam(alpha=args.lr, beta1=args.beta1)
    opt_dis.setup(dis)

    updater = Pix2pixUpdater(models=(gen, dis),
                             iterator={'main': train_iter},
                             optimizer={'gen': opt_gen,
                                        'dis': opt_dis},
                             device=args.gpu)

    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.output_dir)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
        filename="snapshot_iter_{.updater.iteration}.npz"),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'dis/loss']), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # TODO: implement resume

    trainer.run()

if __name__ == '__main__':
    main()
