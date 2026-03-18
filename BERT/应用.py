from model import BERTModel
import torch
import dataset

def try_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda:0')  # 使用第一个GPU
    return torch.device('cpu')

batch_size, max_len = 512, 64
train_iter, vocab = dataset.load_data(batch_size, max_len)

device = try_gpu()

net = BERTModel(len(vocab),num_hiddens=128,norm_shape=[128],ffn_num_input=128,ffn_num_hiddens=256,
                    num_heads=2,num_layers=2,dropout=0.2,key_size=128,query_size=128,value_size=128,
                    hid_in_features=128,mlm_in_features=128,nsp_in_features=128)
net.load_state_dict(torch.load('bert_params.pth', weights_only=True))
net.to(device)

def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = dataset.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=device).unsqueeze(0)
    segments = torch.tensor(segments, device=device).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=device).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X

tokena = ['我','爱','你']
a = get_bert_encoding(net,tokena)
print(a)