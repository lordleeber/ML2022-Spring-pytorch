import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.lstm = nn.LSTM(  # LSTM 效果要比 nn.RNN() 好多了
            input_size=39,  # 图片每行的数据像素点
            hidden_size=hidden_dim,  # rnn hidden unit
            num_layers=10,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
            dropout=0.5,
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        lstm_out, (h_n, h_c) = self.lstm(x, None)  # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值

        # out = self.out(lstm_out[:, -1, :])  - original
        out = self.out(lstm_out.contiguous())
        return out
