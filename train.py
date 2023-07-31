from model import BERTModel
from data import load_data_wiki
import torch

train_loader, vocab = load_data_wiki(batch_size=512, max_len=64)
bert_model = BERTModel(vocab_size=len(vocab),
                       embedding_dim=128,
                       ffn_num_hidden=128,
                       num_heads=2,
                       num_layers=2,
                       dropout=0.2,
                       mlm_hidden=128,
                       nsp_inputs=128,

                       norm_shape=128,
                       )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = torch.nn.CrossEntropyLoss()


def _get_batch_loss_bert(net, loss, vocab_size, tokens_x, segments_x, valid_lens_x, pred_position_x, mlm_weights_x,
                         mlm_y, nsp_y):
    """
    计算一批样本损失，Bert的损失由MLM任务和NSP任务损失之和构成
    :param net:
    :param loss:
    :param vocab_size:
    :param tokens_x: b,seq_len(max_len)
    :param segments_x: b,seq_len(max_len)
    :param valid_lens_x: b
    :param pred_position_x: b,num_mlm_pred(round(max_len*0.15))
    :param mlm_weights_x: b,num_mlm_pred
    :param mlm_y: b,num_mlm_pred
    :param nsp_y: b
    :return:
    """
    _, mlm_y_hat, nsp_y_hat = net(tokens_x, segments_x, valid_lens_x.reshape(-1), pred_position_x)
    mlm_l = loss(mlm_y_hat.reshape(-1, vocab_size), mlm_y.reshape(-1)) * mlm_weights_x.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_x.sum() + 1e-8)
    nsp_l = loss(nsp_y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_loader, net, loss, vocab_size, device, num_steps):
    net = net.to(device)
    optimer = torch.optim.Adam(params=net.parameters(), lr=1e-2)
    num_steps_reached = False
    step = 0
    while step < num_steps and not num_steps_reached:
        for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y) in train_loader:
            tokens_X = tokens_X.to(device)
            segments_X = segments_X.to(device)
            valid_lens_x = valid_lens_x.to(device)
            pred_positions_X = pred_positions_X.to(device)
            mlm_weights_X = mlm_weights_X.to(device)
            mlm_Y = mlm_Y.to(device)
            nsp_y = nsp_y.to(device)
            optimer.zero_grad()
            mlm_l, nsp_l, l = _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                                                   pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            optimer.step()
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break
            print(l.detach().item())


train_bert(train_loader, bert_model, loss_fn, len(vocab), device, 50)
