from collections import OrderedDict
import torch
from torch.nn import Module, TransformerEncoder, TransformerEncoderLayer, Linear, Embedding, Sequential


def get_tokens_and_segents(token_a, token_b=None):
    """获取输入序列的词元及其片段索引
    将原始词元序列转换为BERT输入格式的序列
    :param token_a: 词元, list类型
    :param token_b: 词元, list类型。若为空表示输入为单文本序列
    """
    tokens = ['<cls>'] + token_a + ['<seq>']
    # 标记片段a和片段b
    segments = [0] * (len(tokens))
    if token_b is not None:
        tokens += token_b + ['<seq>']
        segments += [1] * (len(token_b) + 1)
    return tokens, segments


class BERTEncoder(Module):
    def __init__(self, vocab_size, embedding_dim, ffn_num_hidden, num_heads, num_layers,
                 dropout, max_len=1000, norm_shape=None, *args, **kwargs):
        super(BERTEncoder, self).__init__()
        # 对单个词元进行嵌入
        self.token_embedding = Embedding(vocab_size, embedding_dim=embedding_dim)
        # 对段进行嵌入。段只有0或1两种可能
        self.segment_embedding = Embedding(2, embedding_dim=embedding_dim)
        # 对位置进行嵌入. BERT中位置嵌入是本模型的可学习参数，因此构建一个足够长的位置嵌入参数
        self.posi_embedding = torch.nn.Parameter(torch.randn(size=(1, max_len, embedding_dim)))
        # 层归一化
        self.norm = torch.nn.LayerNorm(normalized_shape=norm_shape) if norm_shape is not None else None
        self.encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout,
                                                     dim_feedforward=ffn_num_hidden, batch_first=True, **kwargs)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=self.norm)

    def forward(self, tokens, segaments, valid_lens=None):
        """
        :param tokens: 词元 batch_size,seq_len
        :param segaments: 段 batch_size,seq_len 由0或1组成
        :param valid_lens: 有效长度 batch_size
        :return:
        """
        X = self.token_embedding(tokens) + self.segment_embedding(segaments)
        X += self.posi_embedding.data[:, X.shape[1], :]
        # X.shape=b,seq_len_embedding_dim
        if valid_lens is None:
            X += self.encoder(X)
        else:
            mask = []
            for each in valid_lens:
                mask.append(torch.cat((torch.ones(each.item()), torch.zeros(X.shape[1] - each.item()))))
            mask = 1-torch.stack(mask, dim=0)
            mask = mask.to(tokens.device)
            # mask矩阵中1代表忽略，0代表不忽略，注意！
            X += self.encoder(X, src_key_padding_mask=mask)
        return X


# 下游任务1: MaskLM
class MaskLM(Module):
    def __init__(self, vocab_size, embedding_dim, mlm_hidden, **kwargs):
        super(MaskLM, self).__init__()
        self.mlp = Sequential(OrderedDict({
            "fc1": Linear(in_features=embedding_dim, out_features=mlm_hidden),
            "ac1": torch.nn.ReLU(),
            "norm1": torch.nn.LayerNorm(mlm_hidden),
            "fc2": Linear(in_features=mlm_hidden, out_features=vocab_size)
        }))

    def forward(self, x, pred_position):
        """
        输入BERT编码结果和待预测词元的位置，输出预测位置上的预测结果
        :param x: batch_size,seq_len,embedding_dim
        :param pred_position: batch_size,position_len
        :return: batch_size,position_len,vocab_size
        """
        num_pre_pos = pred_position.shape[1]
        pred_position = pred_position.flatten()
        batch_size = x.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, repeats=num_pre_pos)
        masked_x = x[batch_idx, pred_position]  # 把掩码向量从X中取出来
        # masked_x.shape = batch_size*position_len, embedding_dim
        masked_x = masked_x.reshape([batch_size, num_pre_pos, -1])
        # masked_x.shape = batch_size,position_len,embedding_dim
        # mlp接收掩码向量作为输入，返回每个掩码向量在词汇表上的预测值
        return self.mlp(masked_x)


class NextSentencePredict(Module):
    def __init__(self, nsp_inputs):
        super(NextSentencePredict, self).__init__()
        self.fc = Linear(nsp_inputs, 2)

    def forward(self, X):
        # X.shape=batch_size,num_hiddens
        return self.fc(X)


# 代码整合
class BERTModel(Module):
    def __init__(self, vocab_size, embedding_dim, ffn_num_hidden, num_heads, num_layers,
                 dropout, mlm_hidden, nsp_inputs, max_len=1000, norm_shape=None, *args, **kwargs):
        """

        :param vocab_size:
        :param embedding_dim: Transformer编码器嵌入维度
        :param ffn_num_hidden: Transformer编码器中全连接层隐层维度
        :param num_heads:
        :param num_layers:
        :param dropout:
        :param mlm_hidden: MLM任务中MLP中间隐层神经元数量
        :param nsp_inputs: NSP任务中NSP输入神经元数量，应与embedding_dim保持一致
        :param max_len: Transformer编码器中位置嵌入最大数目，应不小于数据加载器中输入数据最大序列长度
        :param norm_shape: Transformer编码器中LayerNorm参数，与embedding_dim保持一致即可
        :param args:
        :param kwargs:
        """
        super(BERTModel, self).__init__()
        self.bert_encoder = BERTEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                        ffn_num_hidden=ffn_num_hidden, num_heads=num_heads, num_layers=num_layers,
                                        dropout=dropout, max_len=max_len, norm_shape=norm_shape, **kwargs)
        self.mlm = MaskLM(vocab_size=vocab_size, embedding_dim=embedding_dim, mlm_hidden=mlm_hidden)
        self.nsp = NextSentencePredict(nsp_inputs=nsp_inputs)

    def forward(self, tokens, segaments, valid_lens=None, pred_position=None):
        encoded_x = self.bert_encoder(tokens, segaments, valid_lens)  # batch_size,seq_len,embedding_dim
        mlm_y_hat = self.mlm(encoded_x, pred_position) if pred_position is not None else None
        nsp_y_hat = self.nsp(encoded_x[:, 0, :])  # 只用<cls>预测下一句是否为上下文关系
        return encoded_x, mlm_y_hat, nsp_y_hat


if __name__ == '__main__':
    token = torch.randint(low=0, high=1000, size=(2, 8))
    segament = torch.IntTensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    pred_pos = torch.tensor([[1, 2, 5], [1, 5, 6]], dtype=torch.int64)  # (batch_size, pred_pos_len)
    bert_model = BERTModel(vocab_size=10000, embedding_dim=768, ffn_num_hidden=1024, num_heads=4, num_layers=2,
                           dropout=0.2, norm_shape=768, mlm_hidden=256, nsp_inputs=768, max_len=1000)
    bert_model(token, segament, pred_position=pred_pos)
