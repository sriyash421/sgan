import torch
import torch.nn as nn
from .models import make_mlp, get_noise, TrajectoryGenerator, PoolHiddenNet

class ConditionalPoolHiddenNet(PoolHiddenNet):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0
    ):
        super(ConditionalPoolHiddenNet, self).__init__(embedding_dim, h_dim, 
            mlp_dim, bottleneck_dim, activation, batch_norm, dropout)

        mlp_pre_dim = 2 * (embedding_dim + h_dim)
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout)

    def forward(self, h_states, ego_h, ego_end_pos, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - ego_h: ego h per batch
        - ego_end_pos: ego_end_pos per batch
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)

            ###
            # Repeat -> P1', P2', P1', P2'
            ego_h_1 = ego_h[idx].view(-1, self.h_dim).repeat(num_ped, 1).repeat(num_ped, 1)
            ego_end_pos_1 = ego_end_pos[idx].view(-1, 2).repeat(num_ped, 1)
            # Repeat -> P1', P2', P1', P2'
            ego_end_pos_1 = self.spatial_embedding(curr_end_pos - ego_end_pos_1).repeat(num_ped, 1)
            ###
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1, ego_h_1, ego_end_pos_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class CPoolTrajectoryGenerator(TrajectoryGenerator):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
        device='cpu'
    ):
        print("Using CPOOL\n\n\n")
        super(CPoolTrajectoryGenerator, self).__init__(
            obs_len, pred_len, embedding_dim, encoder_h_dim, decoder_h_dim,
            mlp_dim, num_layers, noise_dim, noise_type, noise_mix_type,
            pooling_type, pool_every_timestep, dropout, bottleneck_dim,
            activation, batch_norm, neighborhood_size, grid_size, device
        )
        if pooling_type == 'pool_net':
            self.pool_net = ConditionalPoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, ego_traj, ego_traj_rel, num_samples, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - ego_traj: Tensor of shape (obs_len+future_len, batch, 2) for the ego agent
        - ego_traj_rel: Tensor of shape (obs_len+future_len, batch, 2) for the ego agent
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel) # batch x 1 x hidden_dim
        # Ego Hidden State
        ego_encoder_h = self.encoder(ego_traj_rel).view(-1, self.encoder_h_dim)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, ego_encoder_h, ego_traj[-1, :, :], seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).to(self.device)

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(
            last_pos,
            last_pos_rel,
            state_tuple,
            seq_start_end,
        )
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel

class OptimizedCPool(TrajectoryGenerator):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
        pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8,
        device='cpu'
    ):
        print("Using CPOOL\n\n\n")
        super(OptimizedCPool, self).__init__(
            obs_len, pred_len, embedding_dim, encoder_h_dim, decoder_h_dim,
            mlp_dim, num_layers, noise_dim, noise_type, noise_mix_type,
            pooling_type, pool_every_timestep, dropout, bottleneck_dim,
            activation, batch_norm, neighborhood_size, grid_size, device
        )
        if pooling_type == 'pool_net':
            self.pool_net = ConditionalPoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, ego_traj, ego_traj_rel, num_samples, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - ego_traj: Tensor of shape (obs_len+future_len, batch, 2) for the ego agent
        - ego_traj_rel: Tensor of shape (obs_len+future_len, batch, 2) for the ego agent
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel) # batch x 1 x hidden_dim
        # Ego Hidden State
        ego_encoder_h = self.encoder(ego_traj_rel).view(-1, self.encoder_h_dim)

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, ego_encoder_h, ego_traj[-1, :, :], seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(
            noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(
            self.num_layers, batch, self.decoder_h_dim
        ).to(self.device)

        pred_traj_samples = []

        for _ in range(num_samples):
            # Add Noise
            if self.mlp_decoder_needed():
                noise_input = self.mlp_decoder_context(
                    mlp_decoder_context_input)
            else:
                noise_input = mlp_decoder_context_input
            decoder_h = self.add_noise(
                noise_input, seq_start_end, user_noise=user_noise)
            decoder_h = torch.unsqueeze(decoder_h, 0)

            state_tuple = (decoder_h, decoder_c)
            last_pos = obs_traj[-1]
            last_pos_rel = obs_traj_rel[-1]
            # Predict Trajectory

            pred_traj_samples.append(self.decoder(
                last_pos,
                last_pos_rel,
                state_tuple,
                seq_start_end)[0]
            )
        
        return torch.stack(pred_traj_samples, axis=0)