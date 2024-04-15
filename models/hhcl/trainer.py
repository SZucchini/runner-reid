import time
from logging import getLogger, StreamHandler, DEBUG, Formatter

from .utils.meters import AverageMeter

handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger = getLogger("Log")
logger.setLevel(DEBUG)
for h in logger.handlers[:]:
    logger.removeHandler(h)
    h.close()
logger.addHandler(handler)


class Trainer(object):
    def __init__(self, encoder, memory=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            inputs, labels, indexes = self._parse_data(inputs)

            loss = 0
            f_out = self._forward(inputs)
            loss += self.memory(f_out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.debug('Epoch: [{}][{}/{}]\t'
                             'Time {:.3f} ({:.3f})\t'
                             'Data {:.3f} ({:.3f})\t'
                             'Loss {:.3f} ({:.3f})'
                             .format(epoch, i + 1, train_iters,
                                     batch_time.val, batch_time.avg,
                                     data_time.val, data_time.avg,
                                     losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
