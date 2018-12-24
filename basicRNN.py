import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
import sys
import time
import math
import string


all_letters = ' ' + string.ascii_lowercase + string.ascii_uppercase + '0123456789'
n_letters = len(all_letters)
char_to_index = {char: i for i, char in enumerate(all_letters)}

embedding_dim = 100
batch_size = 16
n_epochs = 10
hidden_size = 128
lr = 0.001


class BasicRNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons, n_outputs):
        super(BasicRNN, self).__init__()

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.E = nn.Embedding(n_letters, embedding_dim, padding_idx=char_to_index[' '])

        self.lstm = nn.LSTM(self.n_inputs * embedding_dim, self.n_neurons, batch_first=True)

        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self):
        # h_0 = (num_layers * num_directions, batch_size, n_neurons)
        # c_0 = (num_layers * num_directions, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons).cuda(),
                torch.zeros(1, self.batch_size, self.n_neurons).cuda())

    def forward(self, X, X_lengths):
        X = X.long()
        X = self.E(X)

        self.batch_size = X.size(0)
        self.hidden = self.init_hidden()

        X = rnn_utils.pack_padded_sequence(X, X_lengths, batch_first=True).cuda()

        X, self.hidden = self.lstm(X, self.hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        X = self.FC(X)

        X = self.softmax(X)
        X = X.view(self.batch_size, max(X_lengths), self.n_outputs)

        # Extract the outputs for the last timestep of each example
        idx = (torch.LongTensor(X_lengths) - 1).view(-1, 1).expand(
            len(X_lengths), X.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension).cuda()

        # Shape: (batch_size, hidden_dim)
        last_output = X.gather(time_dimension, Variable(idx)).squeeze(time_dimension)

        return last_output


model = BasicRNN(batch_size, 1, hidden_size, 2)
model.cuda(0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def main():
    start = time.time()

    train_x, train_y = get_data(sys.argv[1])
    dev_x, dev_y = get_data(sys.argv[2])

    train_loader = to_batch(train_x, train_y)
    dev_loader = to_batch(dev_x, dev_y)

    print('Loading Time: {}'.format(time_since(start)))

    start = time.time()

    for epoch in range(n_epochs):
        print('Epoch:  {}'.format(epoch))

        train_loss = 0.0
        train_acc = 0.0

        for i, i_data in enumerate(train_loader):
            loss, acc = train(i_data)
            train_loss += loss
            train_acc += acc

        print('train:   Loss: {:.4f} | Train Accuracy: {:.2f}'
              .format(train_loss / len(train_x) * 100., train_acc / len(train_x) * 100.))

        dev_loss = 0.0
        dev_acc = 0.0

        for i, i_data in enumerate(dev_loader):
            loss, acc = eval(i_data)
            dev_loss += loss
            dev_acc += acc

        print('dev:   Loss: {:.4f} | Dev Accuracy: {:.2f}\n'
              .format(dev_loss / len(dev_x) * 100., dev_acc / len(dev_x) * 100.))

    print('Overall Time: {}'.format(time_since(start)))


def train(data):
    model.train()
    optimizer.zero_grad()

    model.hidden = model.init_hidden()

    [inputs, input_sizes], labels = data

    outputs = model(inputs, input_sizes)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    loss = loss.item()
    acc = get_accuracy(outputs, labels)

    return loss, acc


def eval(data):
    model.eval()

    model.hidden = model.init_hidden()

    [inputs, input_sizes], labels = data

    outputs = model(inputs, input_sizes)

    loss = criterion(outputs, labels)

    loss = loss.item()
    acc = get_accuracy(outputs, labels)

    return loss, acc


def to_batch(data_x, data_y):
    split_x = []
    split_x_lengths = []
    split_y = []
    for i in range(0, len(data_x), batch_size):
        batch_x = data_x[i:i + batch_size]
        batch_y = data_y[i:i + batch_size]

        combined = list(zip(batch_x, batch_y))
        combined.sort(key=lambda x: len(x[0]), reverse=True)
        batch_x[:], batch_y[:] = zip(*combined)

        split_x.append(pad_tensor_batch(batch_x))
        split_x_lengths.append([len(x) for x in batch_x])
        split_y.append(torch.Tensor(batch_y).long().cuda())

    return list(zip(zip(split_x, split_x_lengths), split_y))


def pad_tensor_batch(batch):
    batch = [torch.Tensor(x).cuda() for x in batch]

    return rnn_utils.pad_sequence(batch, batch_first=True, padding_value=char_to_index[' '])\
        .view(len(batch), -1).cuda()


def get_accuracy(pred, target):
    corrects = (torch.max(pred, 1)[1].view(target.size()).data == target.data).sum()
    return corrects.item()


def get_data(path):
    lines = open(path, encoding='utf-8').read().strip().split('\n')
    data_with_tags = [line.split(' ') for line in lines]
    return [[char_to_index[c] for c in item[0]]
            for item in data_with_tags], [int(item[1]) for item in data_with_tags]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    main()
