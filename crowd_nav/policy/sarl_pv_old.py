import torch
import torch.nn as nn
import numpy as np
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class SARLPV(MultiHumanRL, nn.Module):
    def __init__(self, config):
        MultiHumanRL.__init__(self)
        nn.Module.__init__(self)
        self.name = 'SARL-PV'
        self.global_state_dim = list()
        self.mlp1 = nn.Sequential()
        self.mlp2 = nn.Sequential()
        self.with_global_state = bool()
        self.attention = nn.Sequential()
        self.mlp3 = nn.Sequential()
        self.linear_p_1 = None
        self.linear_p_2 = None
        self.linear_v = None
        self.attention_weights = None
        self.configure(config)

    def configure(self, config):
        """  Configure SARL-PV  object  """
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl_pv', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl_pv', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl_pv', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl_pv', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl_pv', 'with_om')
        with_global_state = config.getboolean('sarl_pv', 'with_global_state')
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(self.input_dim(), mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims[:-1], last_relu=True)
        self.linear_p_2 = nn.Linear(mlp3_dims[-2], mlp3_dims[-1])
        p_d = int(config.get('action_space', 'speed_samples')) * int(config.get('action_space', 'rotation_samples')) + 1
        self.linear_p_1 = nn.Linear(mlp3_dims[-1], p_d)
        self.linear_v = nn.Linear(mlp3_dims[-1], 1)
        self.multiagent_training = config.getboolean('sarl_pv', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL-PV'
        self.phase = 'train'  # Not essential, but keeps observation state

    def get_attention_weights(self):
        return self.attention_weights

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation
        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)
        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)). \
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()
        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        joint_state = self.mlp3(joint_state)
        joint_state = torch.nn.functional.relu(self.linear_p_2(joint_state))
        policies = torch.nn.functional.softmax(self.linear_p_1(joint_state), dim=-1)
        values = self.linear_v(joint_state)
        return policies, values

    def forward_with_processing(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to policy-value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)
        """
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)
        # POLICY, VALUE UPDATE (Look-ahead not appropriate here):
        occupancy_maps = None
        self.action_values = list()
        batch_state = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                 for human_state in state.human_states], dim=0)
        rotated_batch_input = self.rotate(batch_state).unsqueeze(0)
        if self.with_om:
            if occupancy_maps is None:
                occupancy_maps = self.build_occupancy_maps(state.human_states).unsqueeze(0)
            rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
        policy, value = self.forward(rotated_batch_input)
        self.action_values = policy.detach().numpy()[0].tolist()
        if self.phase == 'train':
            self.last_state = self.transform(state)
        return policy, value, np.expand_dims(self.last_state.detach().numpy(), axis=0)
