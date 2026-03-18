import torch
from torch import nn
import dataset
import model

def try_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda:0')  # 使用第一个GPU
    return torch.device('cpu')

def _get_batch_loss_bert(net,loss,vocab_size,tokens_X,segments_X,valid_lens_x,pred_positions_X,mlm_weights_X,mlm_Y,nsp_y):
    _,mlm_Y_hat,nsp_Y_hat= net(tokens_X,segments_X,valid_lens_x.reshape(-1),pred_positions_X)
    mlm_l= loss(mlm_Y_hat.reshape(-1,vocab_size),mlm_Y.reshape(-1))* \
           mlm_weights_X.reshape(-1, 1)
    mlm_l= mlm_l.sum()/(mlm_weights_X.sum()+1e-8)
    nsp_l= loss(nsp_Y_hat,nsp_y)
    l = mlm_l+ nsp_l
    return mlm_l,nsp_l,l

def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = net.to(devices)
    trainer = torch.optim.Adam(net.parameters(), lr=0.0001)
    step_counter = 0  # 初始化步数计数器
    while step_counter < num_steps:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices)
            segments_X = segments_X.to(devices)
            valid_lens_x = valid_lens_x.to(devices)
            pred_positions_X = pred_positions_X.to(devices)
            mlm_weights_X = mlm_weights_X.to(devices)
            mlm_Y, nsp_y = mlm_Y.to(devices), nsp_y.to(devices)
            trainer.zero_grad()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            step_counter += 1
            if step_counter % 100 == 0:
                print(f"步骤 {step_counter}/{num_steps}, MLM损失: {mlm_l.item():.4f}, NSP损失: {nsp_l.item():.4f}")
            if step_counter >= num_steps:
                break

batch_size, max_len = 512, 64
data_address = "data/thuc_no.txt"
train_iter, vocab = dataset.load_data(batch_size, max_len,data_address)
#vocab.load_state_dict(torch.load('vocab.pth'))
net = model.BERTModel(len(vocab),num_hiddens=768,norm_shape=[768],ffn_num_input=768,ffn_num_hiddens=3072,
                    num_heads=12,num_layers=12,dropout=0.1,key_size=768,query_size=768,value_size=768,
                    hid_in_features=768,mlm_in_features=768,nsp_in_features=768)
devices =try_gpu()
loss= nn.CrossEntropyLoss()

train_bert(train_iter, net,loss,len(vocab),devices, 500)
torch.save(vocab.state_dict(), 'vocab.pth')
torch.save(net.state_dict(), 'model_params.pth')