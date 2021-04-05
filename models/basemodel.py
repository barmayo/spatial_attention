from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import norm_col_init, weights_init

from .model_io import ModelOutput


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        target_embedding_sz = args.glove_dim
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(BaseModel, self).__init__()

        self.embed_state = nn.Conv1d(resnet_embedding_sz, 64, 1)
        self.embed_glove = nn.Conv1d(target_embedding_sz, 64, 1)
        self.embed_action = nn.Conv1d(action_space, 64, 1)
        self.embed_memory = nn.Conv1d(hidden_state_sz, 64, 1)

        self.memory_alpha = nn.Conv1d(hidden_state_sz, 3, 1)

        self.lstm = nn.LSTMCell(3136, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_probs, memory, params):

        action_embedding_input = action_probs

        state = state.reshape(-1, 512, 49)
        action = action_embedding_input.unsqueeze(2)
        target = target.unsqueeze(0).unsqueeze(2)
        memory = memory.unsqueeze(2)

        if params is None:
            glove_embedding = self.embed_glove(target)
            action_embedding = self.embed_action(action)
            state_embedding = self.embed_state(state)
            memory_embedding = self.embed_memory(memory)

            alpha = self.memory_alpha(memory)
        else:
            glove_embedding = F.conv1d(target, weight=params["embed_glove.weight"], bias=params["embed_glove.bias"])
            action_embedding = F.conv1d(action, weight=params["embed_action.weight"], bias=params["embed_action.bias"])
            state_embedding = F.conv1d(state, weight=params["embed_state.weight"], bias=params["embed_state.bias"])
            memory_embedding = F.conv1d(memory, weight=params["embed_memory.weight"], bias=params["embed_memory.bias"])

            alpha = F.conv1d(memory, weight=params["memory_alpha.weight"], bias=params["memory_alpha.bias"])

        glove_att = F.normalize(glove_embedding)
        action_att = F.normalize(action_embedding)
        state_att = self.dropout(F.normalize(state_embedding))
        memory_att = F.normalize(memory_embedding)

        theta_ST = state_att.transpose(1, 2).bmm(glove_att).squeeze()
        theta_SM = state_att.transpose(1, 2).bmm(memory_att).squeeze()
        theta_SA = state_att.transpose(1, 2).bmm(action_att).squeeze()

        theta = torch.stack([theta_ST, theta_SM, theta_SA])

        poten = F.softmax((theta.unsqueeze(0).transpose(1, 2).bmm(alpha).squeeze()), dim=0)
        out = (poten * F.relu(state_embedding)).view(-1, 3136)

        return out, out

    def a3clstm(self, embedding, prev_hidden, params):
        if params is None:
            hx, cx = self.lstm(embedding, prev_hidden)
            x = hx
            actor_out = self.actor_linear(x)
            critic_out = self.critic_linear(x)

        else:
            hx, cx = self._backend.LSTMCell(
                embedding,
                prev_hidden,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
            )

            # Change for pytorch 1.01
            # hx, cx = nn._VF.lstm_cell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            x = hx

            critic_out = F.linear(
                x,
                weight=params["critic_linear.weight"],
                bias=params["critic_linear.bias"],
            )
            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        params = model_options.params

        x, image_embedding = self.embedding(state, target, action_probs, hx, params)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding
        )
