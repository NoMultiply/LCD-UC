import os
import sys
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.join(__file__), '../../'))
LOG_ROOT = os.path.join(ROOT, 'exp_data/logs')
DATA_ROOT = os.path.join(ROOT, 'data')
OUTPUT_ROOT = os.path.join(ROOT, 'output')
FIG_ROOT = os.path.join(ROOT, 'figures')


def get_parser():
    parser = argparse.ArgumentParser('Argument Parser for LCD-UC')
    parser.add_argument('--dataset', '-d', type=str, choices=('KuaiRec', 'MovieLens'), required=True)
    parser.add_argument('--box', '-box', action='store_true')
    parser.add_argument('--mask', '-mask', action='store_true')
    parser.add_argument('--attn', '-attn', action='store_true', help='attention')
    parser.add_argument('--device', type=str, choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', '-seed', type=int, default=2023)
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--wd', '-wd', type=float, default=0, help='weight_decay')
    parser.add_argument('--n_hidden', '-dim', type=int, default=64)
    parser.add_argument('--n_hidden_box', '-bdim', type=int, default=64)
    
    parser.add_argument('--ne', '-ne', type=int, default=1000, help='n_epoch')
    parser.add_argument('--bs', '-bs', type=int, default=2048, help='batch_size')
    parser.add_argument('--patience', '-p', type=int, default=5)
    parser.add_argument('--ri', '-ri', type=int, default=50, help='report_interval')

    parser.add_argument('--gd', '-gd', action='store_true', help='gumbel_distribution')
    parser.add_argument('--rec', '-rec', type=float, default=1.0, help='lambda_rec')
    parser.add_argument('--ler', '-ler', type=float, default=0, help='lambda_emb_reg')
    parser.add_argument('--lbr', '-lbr', type=float, default=0, help='lambda_box_reg')
    parser.add_argument('--lmr', '-lmr', type=float, default=0, help='lambda_mask_reg')
    parser.add_argument('--eta_uc', '-eta_uc', type=float, default=0.8)
    parser.add_argument('--eta_box', '-eta_box', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.05)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args, parser


class Logger:
    def __init__(self) -> None:
        self.filename = None
        self.writer = None
        self.start = time.time()

    def get_log_filename(self, args, parser):
        full_filename = '|'.join(map(lambda x: '='.join(
            map(str, x)), args.__dict__.items())) + '.log'
        require_keys = {'dataset', 'base_model', 'box', 'attn', 'mask', 'tag'}
        pairs = []
        for k, v in args.__dict__.items():
            if k not in require_keys and v == parser.get_default(k):
                continue
            pairs.append((k, str(v)))
        filename = '|'.join(map('='.join, pairs)) + '.log'
        return full_filename, filename

    def set_log_file(self, args, parser):
        full_filename, filename = self.get_log_filename(args, parser)
        self.filename = filename
        print(f'[Log File] {filename}\n')
        self.writer = open(os.path.join(LOG_ROOT, filename), 'w')
        self.writer.write(' '.join(sys.argv) + '\n\n')
        self.writer.write('[Full Filename] ' + full_filename + '\n\n')

    def print(self, *values, sep=" ", end="\n", **kwargs):
        print(*values, sep=sep, end=end, **kwargs)
        if self.writer is not None:
            self.writer.write(sep.join(map(str, values)) + end)

    def __del__(self):
        end = time.time()
        self.print(f'\n[Time Cost] {end - self.start}s')
        if self.writer is not None:
            self.writer.close()
            print(f'\n[Log File] {self.filename}')


logger = Logger()


def main():
    args, parser = get_args()
    _, filename = logger.get_log_filename(args, parser)
    print(filename)


if __name__ == '__main__':
    main()
