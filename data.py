from typing import List
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import random
import torchtext
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.datasets import WikiText2
from model import get_tokens_and_segents


def dataset_preprocess(dataset) -> List:
    """
    :param dataset:
    :return: 返回值是一个列表，每个元素为多个句子构成的一维列表
    """
    processed = []
    for paragraph in dataset:
        if paragraph.split(' . ').__len__() >= 2:
            processed.append(paragraph.strip().lower().split(' . '))
    random.shuffle(processed)  # 将段顺序打乱
    return processed


# train_paragraphs = dataset_preprocess(train_set)  # 二维列表，每行(每个元素)代表一段话，每段话至少有2个句子。


def _tokenize(sentence: str) -> List[str]:
    """
    分词工具
    :param sentence:
    :return:
    """
    return sentence.split(' ')  # 默认使用空格分词


def _get_next_sentence(sentence, next_sentence, paragraph):
    if random.random() < 0.5:
        is_next = True
    else:
        is_next = False
        # 首先随机选择一段话，在从一段话中随机选择一个句子
        next_sentence = random.choice(random.choice(paragraph))
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab=None, max_len=1000):
    """

    :param paragraph:
    :param paragraphs:
    :param vocab:
    :param max_len: BERT训练期间输入句子最大长度, 注意不是输入token数量
    :return:
    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(paragraph[i], paragraph[i + 1], paragraphs)
        # 使用空格进行分词
        tokens_a = _tokenize(tokens_a)
        tokens_b = _tokenize(tokens_b)
        # 加上<cls>和2个<seq>长度不应超过最大长度
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segents(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


# _get_nsp_data_from_paragraph(train_paragraphs[0], train_paragraphs, max_len=1000)


def _replace_mlm_tokens(tokens, candicate_pred_positions, num_mlm_preds, vocab: Vocab):
    """
    为遮蔽模型创建输入词元，其中可能包含<mask>或随机词元
    :param tokens: 输入BERT编码器的词元列表
    :param candicate_pred_positions: 不包括特殊词元的BERT随机输入序列词元列表
    :param num_mlm_preds: 指定预测的数量(选择15%预测词元)
    :param vocab:
    :return:
    """
    mlm_input_tokens = copy.deepcopy(tokens)
    pred_position_and_label = []
    # 打算顺序以获得15%随机词元在掩蔽模型中预测
    random.shuffle(candicate_pred_positions)
    for mlm_pred_position in candicate_pred_positions:
        if len(pred_position_and_label) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间将词元替换为<mask>词元
        random_flag = random.random()
        if random_flag < 0.8:
            masked_token = '<mask>'
        elif random_flag < 0.9:
            masked_token = tokens[mlm_pred_position]
        else:
            token_idx = random.randint(0, len(vocab) - 1)
            masked_token = vocab.lookup_token(token_idx)
        # 替换输入序列预测位置词元
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_position_and_label.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_position_and_label


def _get_mlm_data_from_tokens(tokens, vocab: Vocab):
    """

    :param tokens:
    :param vocab:
    :return:
    """
    candicate_pred_positions = []
    for i, token in enumerate(tokens):
        # 对特殊词元不进行预测
        if token in ['<cls>', '<seq>']:
            continue
        candicate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    # 整个序列中除特殊词元之外的词元都是候选预测位置
    # 替换序列中部分词元，替换的位置为预测位置，替换的词元为预测标签
    mlm_input_tokens, pred_position_and_label = _replace_mlm_tokens(
        tokens, candicate_pred_positions, num_mlm_preds, vocab)
    pred_position_and_label = sorted(pred_position_and_label, key=lambda x: x[0])
    pred_position = [each[0] for each in pred_position_and_label]
    mlm_pred_label = [each[1] for each in pred_position_and_label]
    return vocab(mlm_input_tokens), pred_position, vocab(mlm_pred_label)


def _pad_bert_inputs(examples, max_len, vocab: Vocab):
    """
    :param examples: examples为纯数值型列表
    :param max_len:
    :param vocab:
    :return:
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        # 使用<pad>对词元索引列表进行填充补齐
        all_token_ids.append(torch.tensor(token_ids + vocab(['<pad>']) * (max_len - len(token_ids)), dtype=torch.long))
        # 用0填充段嵌入列表
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括<pad>的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.int))
        # 用0填充预测位置
        all_pred_positions.append(
            torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测通过乘以0权重在损失中过滤
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)),
                         dtype=torch.float32))
        all_mlm_labels.append(
            torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels


# 出现次数不少于5次的不频繁词元将被过滤
class _WikiTextDataset(Dataset):
    def __init__(self, paragraphs, max_len):
        """
        :param paragraphs: 二维列表，每行由多个句子组成
        :param max_len:
        """
        super(_WikiTextDataset, self).__init__()
        sentences = [_tokenize(sentence) for paragraph in paragraphs for sentence in paragraph]
        self.vocab = build_vocab_from_iterator(sentences, min_freq=5,
                                               specials=['<pad>', '<mask>', '<cls>', '<seq>', '<unk>'])
        self.vocab.set_default_index(4)
        # 20256个词元
        # 构造nsp任务数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
            # 在这一步，词元数量超过max_len的句子不会被保留
        # 构造mlm任务数据
        examples = [_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next) for tokens, segments, is_next in
                    examples]
        # examples的元素为(input_tokens, pred_position, pred_label, segments, is_next)
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, item):
        return self.all_token_ids[item], self.all_segments[item], self.valid_lens[item], self.all_pred_positions[item], \
            self.all_mlm_weights[item], \
            self.all_mlm_labels[item], self.nsp_labels[item]

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    data_dir = './'
    train_set = WikiText2(root=data_dir, split='train')
    valid_set = WikiText2(root=data_dir, split='valid')
    test_set = WikiText2(root=data_dir, split='test')
    train_paragraphs = dataset_preprocess(train_set)
    valid_paragraphs = dataset_preprocess(valid_set)
    test_paragraphs = dataset_preprocess(test_set)
    train_wikiset = _WikiTextDataset(train_paragraphs, max_len)
    train_loader = DataLoader(train_wikiset, batch_size=batch_size, shuffle=True, drop_last=True)
    print('train set size:', len(train_wikiset))
    return train_loader, train_wikiset.vocab


if __name__ == '__main__':
    train_loader, _ = load_data_wiki(batch_size=512, max_len=64)
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) in train_loader:
        print(tokens_X.shape, segments_X.shape, valid_lens_x.shape, pred_positions_X.shape, mlm_weights_X.shape,
              mlm_Y.shape, nsp_y.shape)
        break
