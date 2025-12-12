import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange
import math


class Model(nn.Module):
    """
    H²RL: Heterogeneous Hypergraph Representation Learning
    Combines HyperIMTS hypergraph architecture with heterogeneous graph learning
    for time series representation learning
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.configs = configs

        # Multi-Domain Hypergraph Construction
        self.multi_domain_encoder = MultiDomainHypergraphEncoder(
            seq_len=configs.seq_len,
            enc_in=configs.enc_in,
            d_model=configs.d_model,
            n_freq_modes=configs.d_model // 4  # number of frequency components
        )

        # Heterogeneous Hypergraph Learning Layers
        self.hypergraph_layers = nn.ModuleList([
            HeterogeneousHypergraphLayer(
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                dropout=configs.dropout
            ) for _ in range(configs.e_layers)
        ])

        # Meta-Network for Dynamic Adaptation
        self.meta_network = MetaAdaptationNetwork(
            d_model=configs.d_model,
            n_hyperedge_types=3  # temporal, frequency, cross-domain
        )

        # Structure-based Contrastive Learning (no augmentation)
        self.structural_contrastive = StructuralContrastiveLearning(
            d_model=configs.d_model,
            temperature=configs.temperature if hasattr(configs, 'temperature') else 0.2
        )

        # Reconstruction heads
        if self.task_name == 'pretrain':
            self.reconstruction_head = nn.Linear(3 * configs.d_model, 1)
            self.mse = nn.MSELoss()
        elif self.task_name == 'finetune':
            self.forecast_head = ForecastHead(
                seq_len=configs.seq_len,
                d_model=configs.d_model,
                pred_len=configs.pred_len,
                enc_in=configs.enc_in  # ADD THIS PARAMETER
            )

        self.layer_norm = nn.LayerNorm(configs.d_model)

    def forward(self, x_enc, x_mark_enc, batch_x=None, mask=None):
        if self.task_name == 'pretrain':
            return self.pretrain(x_enc, x_mark_enc, batch_x, mask)
        elif self.task_name == 'finetune':
            return self.forecast(x_enc, x_mark_enc)
        return None

    def pretrain(self, x_enc, x_mark_enc, batch_x, mask):
        """Pre-training with structure-based contrastive learning"""
        bs, seq_len, n_vars = x_enc.shape

        # Normalization with masking
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # Multi-Domain Heterogeneous Hypergraph Construction
        (
            time_nodes,
            freq_nodes,
            stat_nodes,
            temporal_hyperedges,
            frequency_hyperedges,
            cross_domain_hyperedges,
            incidence_matrices
        ) = self.multi_domain_encoder(x_enc, mask)

        # Meta-learning for adaptive hyperedge weighting
        hyperedge_weights = self.meta_network(x_enc, mask)

        # Heterogeneous Hypergraph Message Passing
        for i, layer in enumerate(self.hypergraph_layers):
            time_nodes, freq_nodes, stat_nodes, temporal_hyperedges, frequency_hyperedges, cross_domain_hyperedges = layer(
                time_nodes=time_nodes,
                freq_nodes=freq_nodes,
                stat_nodes=stat_nodes,
                temporal_hyperedges=temporal_hyperedges,
                frequency_hyperedges=frequency_hyperedges,
                cross_domain_hyperedges=cross_domain_hyperedges,
                incidence_matrices=incidence_matrices,
                hyperedge_weights=hyperedge_weights,
                mask=mask
            )

        # Structure-based Contrastive Learning (no augmentation)
        series_repr = self.aggregate_representations(time_nodes, freq_nodes, stat_nodes)
        loss_cl, similarity_matrix = self.structural_contrastive(
            series_repr,
            incidence_matrices,
            mask
        )

        # Reconstruction using weighted aggregation
        # similarity_matrix: (bs, bs) where bs might be 2*batch_size due to augmentation
        actual_bs = similarity_matrix.shape[0]
        
        # Create rebuild weight matrix
        rebuild_weight_matrix = torch.softmax(similarity_matrix / self.configs.temperature, dim=-1)
        rebuild_weight_matrix = rebuild_weight_matrix - torch.eye(rebuild_weight_matrix.shape[0]).to(x_enc.device) * 1e12
        rebuild_weight_matrix = torch.softmax(rebuild_weight_matrix / self.configs.temperature, dim=-1)
        
        # Aggregate multi-domain representations for reconstruction
        # time_nodes: (bs, seq_len, n_vars, d_model)
        # We need to flatten to (bs, seq_len * n_vars, d_model) then aggregate
        
        # Reshape nodes to (actual_bs, seq_len, n_vars, d_model)
        time_nodes_flat = time_nodes.reshape(actual_bs, seq_len, n_vars, self.d_model)
        
        # Average across d_model to get (actual_bs, seq_len, n_vars)
        time_repr_spatial = time_nodes_flat.mean(dim=-1)  # (actual_bs, seq_len, n_vars)
        
        # Apply similarity-based reconstruction
        # Reshape for batch-wise reconstruction
        time_repr_flat = time_repr_spatial.reshape(actual_bs, -1)  # (actual_bs, seq_len*n_vars)
        
        # Weighted aggregation across batch
        reconstructed_flat = torch.matmul(rebuild_weight_matrix, time_repr_flat)  # (actual_bs, seq_len*n_vars)
        
        # Reshape back
        dec_out = reconstructed_flat.reshape(actual_bs, seq_len, n_vars)

        # De-normalization
        dec_out = dec_out * stdev
        dec_out = dec_out + means

        # Use only the first batch_x.shape[0] samples for reconstruction loss
        pred_batch_x = dec_out[:batch_x.shape[0]]
        loss_rb = self.mse(pred_batch_x, batch_x.detach())

        # Combined loss
        loss = loss_cl + loss_rb

        return loss, loss_cl, loss_rb, None, similarity_matrix, rebuild_weight_matrix, pred_batch_x

    def forecast(self, x_enc, x_mark_enc):
        """Forecasting using learned hypergraph representations"""
        bs, seq_len, n_vars = x_enc.shape

        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Multi-Domain Hypergraph Construction
        mask = torch.ones_like(x_enc)
        (
            time_nodes,
            freq_nodes,
            stat_nodes,
            temporal_hyperedges,
            frequency_hyperedges,
            cross_domain_hyperedges,
            incidence_matrices
        ) = self.multi_domain_encoder(x_enc, mask)

        # Meta-adaptation
        hyperedge_weights = self.meta_network(x_enc, mask)

        # Message passing
        for layer in self.hypergraph_layers:
            time_nodes, freq_nodes, stat_nodes, temporal_hyperedges, frequency_hyperedges, cross_domain_hyperedges = layer(
                time_nodes, freq_nodes, stat_nodes,
                temporal_hyperedges, frequency_hyperedges, cross_domain_hyperedges,
                incidence_matrices, hyperedge_weights, mask
            )

        # Aggregate representations
        time_repr = time_nodes.mean(dim=(1, 2))  # (bs, d_model)
        freq_repr = freq_nodes.mean(dim=(1, 2))  # (bs, d_model)
        stat_repr = stat_nodes.mean(dim=(1, 2))  # (bs, d_model)

        combined_repr = torch.cat([time_repr, freq_repr, stat_repr], dim=-1)  # (bs, 3*d_model)

        # Forecast
        dec_out = self.forecast_head(combined_repr)  # (bs, pred_len, n_vars)

        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def aggregate_representations(self, time_nodes, freq_nodes, stat_nodes):
        """Aggregate multi-domain node representations for series-level representation"""
        # time_nodes: (bs, seq_len, enc_in, d_model)
        # freq_nodes: (bs, n_freq_modes, enc_in, d_model)
        # stat_nodes: (bs, 3, enc_in, d_model)
        
        # Global pooling across sequence and variable dimensions for each domain
        time_repr = time_nodes.mean(dim=(1, 2))  # (bs, d_model)
        freq_repr = freq_nodes.mean(dim=(1, 2))  # (bs, d_model)
        stat_repr = stat_nodes.mean(dim=(1, 2))  # (bs, d_model)
        
        # Concatenate all domains
        return torch.cat([time_repr, freq_repr, stat_repr], dim=-1)  # (bs, 3*d_model)


class MultiDomainHypergraphEncoder(nn.Module):
    """
    Constructs heterogeneous hypergraph with multiple node types:
    - Time-domain nodes (raw observations)
    - Frequency-domain nodes (FFT components)
    - Statistical nodes (summary statistics)
    """

    def __init__(self, seq_len, enc_in, d_model, n_freq_modes):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.n_freq_modes = n_freq_modes

        # Node encoders
        self.time_node_encoder = nn.Linear(1, d_model)
        self.freq_node_encoder = nn.Linear(2, d_model)  # real and imaginary parts
        self.stat_node_encoder = nn.Linear(3, d_model)  # mean, variance, trend

        # Hyperedge encoders
        self.temporal_hyperedge_encoder = nn.Linear(d_model, d_model)
        self.frequency_hyperedge_encoder = nn.Linear(d_model, d_model)
        self.cross_domain_hyperedge_encoder = nn.Linear(d_model, d_model)

        self.activation = nn.ReLU()

    def forward(self, x, mask):
        """
        x: (bs, seq_len, enc_in)
        mask: (bs, seq_len, enc_in)
        """
        bs, seq_len, enc_in = x.shape

        # === TIME-DOMAIN NODES ===
        time_nodes = self.time_node_encoder(x.unsqueeze(-1))  # (bs, seq_len, enc_in, d_model)
        time_nodes = self.activation(time_nodes)

        # === FREQUENCY-DOMAIN NODES ===
        # Apply FFT to extract frequency components
        x_fft = torch.fft.rfft(x, dim=1)  # (bs, seq_len//2+1, enc_in)
        
        # Select top-k frequency modes
        freq_magnitude = torch.abs(x_fft)
        topk_indices = torch.topk(freq_magnitude.mean(dim=-1), self.n_freq_modes, dim=1).indices
        
        # Extract real and imaginary parts
        freq_features = torch.zeros(bs, self.n_freq_modes, enc_in, 2).to(x.device)
        for b in range(bs):
            for i, idx in enumerate(topk_indices[b]):
                freq_features[b, i, :, 0] = x_fft[b, idx, :].real
                freq_features[b, i, :, 1] = x_fft[b, idx, :].imag
        
        freq_nodes = self.freq_node_encoder(freq_features)  # (bs, n_freq_modes, enc_in, d_model)
        freq_nodes = self.activation(freq_nodes)

        # === STATISTICAL NODES ===
        # Compute statistics per variable
        masked_x = x * mask
        n_obs = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (bs, 1, enc_in)
        
        mean_stats = (masked_x.sum(dim=1) / n_obs.squeeze(1)).unsqueeze(1)  # (bs, 1, enc_in)
        var_stats = ((masked_x - mean_stats) ** 2 * mask).sum(dim=1, keepdim=True) / n_obs
        
        # Simple trend estimation (linear regression coefficient)
        time_indices = torch.arange(seq_len).float().to(x.device).view(1, -1, 1)
        trend_stats = ((masked_x * time_indices * mask).sum(dim=1, keepdim=True) / n_obs - 
                    mean_stats * time_indices.mean())
        
        stat_features = torch.cat([mean_stats, var_stats, trend_stats], dim=1)  # (bs, 3, enc_in)
        
        # FIX: Correct reshaping for stat_node_encoder
        # stat_features shape: (bs, 3, enc_in)
        # We need to reshape to (bs, enc_in, 3) then encode to (bs, enc_in, d_model)
        # Then reshape back to (bs, 3, enc_in, d_model) to match other node types
        
        stat_features_reshaped = stat_features.transpose(1, 2)  # (bs, enc_in, 3)
        stat_nodes = self.stat_node_encoder(stat_features_reshaped)  # (bs, enc_in, d_model)
        
        # Reshape to match expected output: (bs, 3, enc_in, d_model)
        # We have one representation per variable, but we need 3 statistical node types
        # Solution: repeat across the first dimension to create 3 statistical perspectives
        stat_nodes = stat_nodes.unsqueeze(1).repeat(1, 3, 1, 1)  # (bs, 3, enc_in, d_model)
        stat_nodes = self.activation(stat_nodes)

        # === HYPEREDGE CONSTRUCTION ===
        # Temporal hyperedges: connect observations at same timestamp
        temporal_hyperedges = self.temporal_hyperedge_encoder(
            time_nodes.mean(dim=2)
        )  # (bs, seq_len, d_model)
        
        # Frequency hyperedges: connect nodes with similar frequency patterns
        frequency_hyperedges = self.frequency_hyperedge_encoder(
            freq_nodes.mean(dim=2)
        )  # (bs, n_freq_modes, d_model)
        
        # Cross-domain hyperedges: bridge different domains
        cross_domain_hyperedges = self.cross_domain_hyperedge_encoder(
            torch.cat([
                time_nodes.mean(dim=(1, 2)),
                freq_nodes.mean(dim=(1, 2)),
                stat_nodes.mean(dim=(1, 2))
            ], dim=-1).view(bs, 3, -1)
        )  # (bs, 3, d_model)

        # Incidence matrices
        incidence_matrices = self.construct_incidence_matrices(bs, seq_len, enc_in, mask)

        return (
            time_nodes,
            freq_nodes,
            stat_nodes,
            temporal_hyperedges,
            frequency_hyperedges,
            cross_domain_hyperedges,
            incidence_matrices
        )

    def construct_incidence_matrices(self, bs, seq_len, enc_in, mask):
        """Construct incidence matrices for hypergraph message passing"""
        # Temporal incidence: (bs, seq_len, seq_len*enc_in)
        temporal_incidence = torch.zeros(bs, seq_len, seq_len * enc_in).to(mask.device)
        for t in range(seq_len):
            temporal_incidence[:, t, t*enc_in:(t+1)*enc_in] = mask[:, t, :]
        
        return {
            'temporal': temporal_incidence,
            'mask': mask
        }


class HeterogeneousHypergraphLayer(nn.Module):
    """
    Heterogeneous hypergraph message passing layer with:
    - Node-to-hyperedge aggregation
    - Hyperedge-to-hyperedge communication
    - Hyperedge-to-node propagation
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Node-to-hyperedge attention
        self.node2hyperedge_time = MultiHeadHypergraphAttention(d_model, n_heads, dropout)
        self.node2hyperedge_freq = MultiHeadHypergraphAttention(d_model, n_heads, dropout)
        
        # Hyperedge-to-hyperedge communication
        self.hyperedge2hyperedge = IrregularityAwareAttention(d_model)
        
        # Hyperedge-to-node propagation
        self.hyperedge2node = nn.Linear(3 * d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, time_nodes, freq_nodes, stat_nodes,
                temporal_hyperedges, frequency_hyperedges, cross_domain_hyperedges,
                incidence_matrices, hyperedge_weights, mask):
        
        # Node-to-hyperedge message passing
        temporal_hyperedges_new = self.node2hyperedge_time(
            temporal_hyperedges,
            time_nodes.reshape(time_nodes.shape[0], -1, self.d_model),
            incidence_matrices['temporal']
        )
        
        # Hyperedge-to-hyperedge communication
        all_hyperedges = torch.cat([
            temporal_hyperedges_new.mean(dim=1, keepdim=True),
            frequency_hyperedges.mean(dim=1, keepdim=True),
            cross_domain_hyperedges.mean(dim=1, keepdim=True)
        ], dim=1)  # (bs, 3, d_model)
        
        all_hyperedges_new = self.hyperedge2hyperedge(
            all_hyperedges,
            adjacency_mask=None
        )
        
        # Hyperedge-to-node propagation
        time_nodes_new = self.hyperedge2node(
            torch.cat([
                time_nodes,
                all_hyperedges_new[:, 0:1, :].unsqueeze(2).expand(-1, time_nodes.shape[1], time_nodes.shape[2], -1),
                all_hyperedges_new[:, 1:2, :].unsqueeze(2).expand(-1, time_nodes.shape[1], time_nodes.shape[2], -1)
            ], dim=-1)
        )
        
        time_nodes = self.layer_norm(time_nodes + self.dropout(self.activation(time_nodes_new)))
        
        return time_nodes, freq_nodes, stat_nodes, temporal_hyperedges_new, frequency_hyperedges, all_hyperedges_new


class MultiHeadHypergraphAttention(nn.Module):
    """Multi-head attention for hypergraph message passing"""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, incidence_matrix):
        """
        queries: (bs, n_hyperedges, d_model)
        keys: (bs, n_nodes, d_model)
        incidence_matrix: (bs, n_hyperedges, n_nodes)
        """
        bs = queries.shape[0]

        Q = self.q_linear(queries).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(keys).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(keys).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply incidence matrix as mask
        if incidence_matrix is not None:
            mask = incidence_matrix.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        out = self.out_linear(out)

        return out


class IrregularityAwareAttention(nn.Module):
    """
    Attention mechanism that adapts to irregularity in time series
    (inspired by HyperIMTS)
    """

    def __init__(self, d_model):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x, adjacency_mask=None):
        """
        x: (bs, n_variables, d_model)
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if adjacency_mask is not None:
            scores = scores.masked_fill(adjacency_mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return out


class MetaAdaptationNetwork(nn.Module):
    """
    Meta-learning network for adaptive hyperedge weighting
    based on data characteristics (non-stationarity handling)
    """

    def __init__(self, d_model, n_hyperedge_types):
        super().__init__()
        self.d_model = d_model
        self.n_hyperedge_types = n_hyperedge_types

        # Statistical feature extraction
        self.stat_encoder = nn.Sequential(
            nn.Linear(4, 32),  # mean, var, spectral_entropy, trend
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Meta-network for hyperedge weights
        self.meta_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_hyperedge_types),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, mask):
        """
        x: (bs, seq_len, enc_in)
        mask: (bs, seq_len, enc_in)
        Returns: (bs, n_hyperedge_types) weights
        """
        bs, seq_len, enc_in = x.shape

        # Compute statistical features
        masked_x = x * mask
        n_obs = mask.sum(dim=(1, 2), keepdim=False).clamp(min=1)  # (bs,)

        # Mean feature: average across all timesteps and variables
        mean_feat = masked_x.sum(dim=(1, 2)) / n_obs  # (bs,)
        mean_feat = mean_feat.unsqueeze(-1)  # (bs, 1)
        
        # Variance feature: compute variance properly
        # First get mean per sample for broadcasting
        mean_broadcast = (masked_x.sum(dim=(1, 2), keepdim=True) / n_obs.view(bs, 1, 1))  # (bs, 1, 1)
        var_feat = (((masked_x - mean_broadcast) ** 2 * mask).sum(dim=(1, 2)) / n_obs).unsqueeze(-1)  # (bs, 1)
        
        # Simple spectral entropy approximation
        x_fft = torch.fft.rfft(x, dim=1)  # (bs, freq_bins, enc_in)
        power_spectrum = torch.abs(x_fft) ** 2
        power_spectrum_sum = power_spectrum.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (bs, 1, enc_in)
        power_spectrum = power_spectrum / power_spectrum_sum
        spectral_entropy = -(power_spectrum * torch.log(power_spectrum + 1e-8)).sum(dim=1).mean(dim=1, keepdim=True)  # (bs, 1)
        
        # Trend (simple linear coefficient)
        time_indices = torch.arange(seq_len).float().to(x.device).view(1, -1, 1)  # (1, seq_len, 1)
        time_mean = time_indices.mean()
        trend_feat = ((masked_x * time_indices * mask).sum(dim=(1, 2)) / n_obs).unsqueeze(-1)  # (bs, 1)

        # Concatenate all statistical features
        stat_features = torch.cat([mean_feat, var_feat, spectral_entropy, trend_feat], dim=-1)  # (bs, 4)

        # Get hyperedge weights
        encoded = self.stat_encoder(stat_features)  # (bs, 64)
        weights = self.meta_net(encoded)  # (bs, n_hyperedge_types)

        return weights


class StructuralContrastiveLearning(nn.Module):
    """
    Structure-based contrastive learning without augmentation
    Uses hypergraph structure to determine similarity
    """

    def __init__(self, d_model, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)
        )

    def forward(self, series_repr, incidence_matrices, mask):
        """
        series_repr: (bs, 3*d_model) - concatenated multi-domain representations
        incidence_matrices: dict containing hypergraph structure info
        mask: (bs, seq_len, enc_in) - masking pattern
        """
        # Ensure series_repr is 2D
        if series_repr.dim() > 2:
            series_repr = series_repr.reshape(series_repr.shape[0], -1)
        
        bs = series_repr.shape[0]
        
        # Project to contrastive space
        z = self.projection(series_repr)  # (bs, 128)
        z = F.normalize(z, dim=1)

        # Compute similarity matrix
        # For 2D tensor, use transpose or matrix multiplication with transposed version
        similarity_matrix = torch.matmul(z, z.transpose(0, 1)) / self.temperature  # (bs, bs)

        # Structure-derived positives: based on hypergraph connectivity
        # Samples with similar masking patterns are considered positive
        mask_flat = mask.reshape(mask.shape[0], -1)  # (bs, seq_len*enc_in)
        mask_norm = mask_flat.sum(dim=1, keepdim=True).clamp(min=1)  # (bs, 1)
        
        # Compute mask similarity using normalized dot product
        mask_similarity = torch.matmul(mask_flat, mask_flat.transpose(0, 1)) / (mask_norm * mask_norm.transpose(0, 1)).clamp(min=1)  # (bs, bs)
        
        # Threshold for positive pairs
        positive_threshold = 0.7
        positives_mask = (mask_similarity > positive_threshold).float()
        positives_mask.fill_diagonal_(0)  # Remove self-similarity

        # Check if there are any positive pairs
        has_positives = positives_mask.sum() > 0

        if has_positives:
            # Contrastive loss
            exp_sim = torch.exp(similarity_matrix)
            exp_sim_diag_removed = exp_sim - torch.diag(torch.diag(exp_sim))
            
            # Positive and negative terms
            pos_sim = (exp_sim * positives_mask).sum(dim=1)
            neg_sim = exp_sim_diag_removed.sum(dim=1)
            
            # Loss: -log(pos / (pos + neg))
            # Add small epsilon to avoid log(0) and division by zero
            loss = -torch.log((pos_sim + 1e-8) / (neg_sim + 1e-8)).mean()
        else:
            # If no positive pairs found, use a simpler contrastive approach
            # Encourage diversity in representations
            exp_sim = torch.exp(similarity_matrix)
            
            # Normalize to create a pseudo loss (push representations apart)
            loss = torch.log(exp_sim.sum(dim=1)).mean()

        return loss, similarity_matrix


class ForecastHead(nn.Module):
    """Forecasting head for final prediction"""

    def __init__(self, seq_len, d_model, pred_len, enc_in):
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.fc = nn.Sequential(
            nn.Linear(3 * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, pred_len * enc_in)
        )

    def forward(self, x):
        """
        x: (bs, 3*d_model)
        Returns: (bs, pred_len, enc_in)
        """
        out = self.fc(x)  # (bs, pred_len * enc_in)
        out = out.reshape(x.shape[0], self.pred_len, self.enc_in)  # (bs, pred_len, enc_in)
        return out
