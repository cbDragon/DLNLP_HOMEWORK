import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import jieba

# # Words to index map
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __len__(self):
        return len(self.word2idx)
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx  # Map to word
            self.idx2word[self.idx] = word  # Map to index
            self.idx += 1

# # Data set part
class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size, device):
        # step 1 Load the data and generate the dictionary
        tokens = 0
        stop_words = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '新语丝电子文库',
                  '\u3000', '\n', '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']
        print(path)
        with open(path, 'r', encoding="ANSI") as f:
            for line in f.readlines():
                # Delete the meaningless stop words
                for sw in stop_words:
                    line = line.replace(sw, '')
                # jieba
                words = jieba.lcut(line) + ['<eos>']
                tokens += len(words)
                # Create a word to index both map
                for word in words:
                    self.dictionary.add_word(word)
        # step 2 Generate a long tensor to store the whole dataSet
        ids = torch.LongTensor(tokens).to(device)
        token = 0
        with open(path, 'r', encoding="ANSI") as f:
            for line in f.readlines():
                for sw in stop_words:
                    line = line.replace(sw, '')
                words = jieba.lcut(line) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        # step 3 Re-construct a matrix according to different batches
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches * batch_size]
        ids = ids.view(batch_size, -1)
        return ids

# # Model part
class LSTMmodel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMmodel, self).__init__()
        drop_out = 0.5
        self.embed = nn.Embedding(vocab_size, embed_size)  # Embed the words
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.dropout = nn.Dropout(drop_out)  # Dropout layer
        self.linear = nn.Linear(hidden_size, vocab_size)  # FC layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid layer--no use

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.dropout(out)
        out = self.linear(out)
        return out, (h, c)

if __name__ == '__main__':
    # # Training Parameters
    embed_size = 256  # Decide each word's characteristic
    hidden_size = 1024  # Decide LSTM hidden node numbers
    num_layers = 4  # Decide LSTM layers
    num_epochs = 30  # Decide training times
    batch_size = 50
    seq_length = 30  # Decide forecast sliding window
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus = Corpus()  # Load training data
    ids = corpus.get_data("dataSet/侠客行.txt", batch_size, device)  # Get data set
    vocab_size = len(corpus.dictionary)  # Number of the whole dictionary with different words

    flag = 0

    if flag:
        model = LSTMmodel(vocab_size, embed_size, hidden_size, num_layers).to(device)

        cost = nn.CrossEntropyLoss()  # Decide the loss function type
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                      torch.zeros(num_layers, batch_size, hidden_size).to(device))  # Initiate the hidden parameters
            print("This is the ", epoch, " epoch")
            for i in range(0, ids.size(1) - seq_length, seq_length):
                inputs = ids[:, i:i + seq_length].to(device)  # Input of train set
                targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)  # Label of train set

                states = [state.detach() for state in states]  # Cut the broadcast of backward gradiant
                outputs, states = model(inputs, states)
                loss = cost(outputs, targets.reshape(-1))  # Calculate the loss function use CrossEntropy

                # Make the model grad into zero,use local update
                model.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=0.5)  # Avoid grad explosion
                optimizer.step()
        save_path = './model_path/model_xkx.pt'  # save model
        torch.save(model, save_path)
    else:
        model = torch.load('./model_path/model_xkx_ep30_2.pt')  # load model

    # # Generate doc
    num_samples = 500  # Length of generate txt
    article = str()  # Store the generative txt in string style

    # Decide the guidance word
    input_para = '开封东门十二里处，有个小市镇，叫做侯监集。这小镇便因侯嬴而得名。'
    input_words = jieba.lcut(input_para)
    print("Guidance words:", input_words)
    input_len = len(input_words)
    input_lst = []
    # Turn the input string word into index type
    for input_word in input_words:
        lst = [corpus.dictionary.word2idx[input_word]]
        input_lst.append(lst)
    _input = torch.Tensor(input_lst).to(device).to(dtype=torch.long)
    # Initiate the (h, c) parameters and output prob
    state = (torch.zeros(num_layers, input_len, hidden_size).to(device),
             torch.zeros(num_layers, input_len, hidden_size).to(device))  # Init parameters of (h, c)
    prob = torch.zeros(vocab_size)  # Init words category probability
    article = ''.join(input_para)

    # # Generate Word after Word
    for i in range(num_samples):
        output, state = model(_input, state)
        # Get the max prob output category with random choice
        prob = output.exp()
        word_id = torch.multinomial(prob, num_samples=1)
        # Get the last output id of the forecast list words
        for j in word_id:
            word_value = j.item()
        # Use input[1:] + new_world to generate the new input
        word_tensor = torch.Tensor([word_value]).to(device).to(dtype=torch.long)
        # Squeeze the tensor to drop the 1 dimension
        _input_squeeze = _input.squeeze()
        # Find the sequence back n-1 word and plus the new forecast word to create a new sequence
        _input = _input_squeeze[1:]
        _input = torch.cat((_input, word_tensor), 0).unsqueeze(1).to(dtype=torch.long)
        # Generate the current forecast word
        word = corpus.dictionary.idx2word[word_value]
        article += word
    print(article)