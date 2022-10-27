import os
import random
from pathlib import Path
from hw5_utils import *
import sentencepiece as spm


data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
urls = (
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted2020.tgz",
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/test.tgz",
)
file_names = (
    'ted2020.tgz', # train & dev
    'test.tgz', # test
)
prefix = Path(data_dir).absolute() / dataset_name


"""## Language"""

src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'


"""## Preprocess files"""

clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)


"""## Split into train/valid"""

valid_ratio = 0.01 # 3000~4000 would suffice
train_ratio = 1 - valid_ratio

if (prefix/f'train.clean.{src_lang}').exists() and (prefix/f'train.clean.{tgt_lang}').exists() \
and (prefix/f'valid.clean.{src_lang}').exists() and (prefix/f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')
else:
    line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))
    labels = list(range(line_num))
    random.shuffle(labels)
    for lang in [src_lang, tgt_lang]:
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')
        count = 0
        for line in open(f'{data_prefix}.clean.{lang}', 'r'):
            if labels[count]/line_num < train_ratio:
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()


"""## Subword Units 
Out of vocabulary (OOV) has been a major problem in machine translation. This can be alleviated by using subword units.
- We will use the [sentencepiece](#kudo-richardson-2018-sentencepiece) package
- select 'unigram' or 'byte-pair encoding (BPE)' algorithm
"""


vocab_size = 8000
if (prefix/f'spm{vocab_size}.model').exists():
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(
        input=','.join([f'{prefix}/train.clean.{src_lang}',
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}',
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 'bpe' works as well
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))
in_tag = {
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:
    for lang in [src_lang, tgt_lang]:
        out_path = prefix/f'{split}.{lang}'
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix/f'{split}.{lang}', 'w') as out_f:
                with open(prefix/f'{in_tag[split]}.{lang}', 'r') as in_f:
                    for line in in_f:
                        line = line.strip()
                        tok = spm_model.encode(line, out_type=str)
                        print(' '.join(tok), file=out_f)

# !head {data_dir+'/'+dataset_name+'/train.'+src_lang} -n 5
# !head {data_dir+'/'+dataset_name+'/train.'+tgt_lang} -n 5

"""## Binarize the data with fairseq"""

binpath = Path('./DATA/data-bin', dataset_name)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    print("waiting execute following command")
    # python -m fairseq_cli.preprocess --source-lang en --target-lang zh --trainpref /home/user/Documents/github/ML2022-Spring-pytorch/HW05/DATA/rawdata/ted2020/train --validpref /home/user/Documents/github/ML2022-Spring-pytorch/HW05/DATA/rawdata/ted2020/valid --testpref /home/user/Documents/github/ML2022-Spring-pytorch/HW05/DATA/rawdata/ted2020/test --destdir DATA/data-bin/ted2020 --joined-dictionary --workers 2
    print(f"--source-lang {src_lang} --target-lang {tgt_lang} --trainpref {prefix/'train'} --validpref {prefix/'valid'} --testpref {prefix/'test'} --destdir {binpath} --joined-dictionary --workers 2")
    # !python -m fairseq_cli.preprocess \
    #       --source-lang {src_lang} --target-lang {tgt_lang} --trainpref {prefix/'train'} --validpref {prefix/'valid'} --testpref {prefix/'test'} --destdir {binpath} --joined-dictionary --workers 2

