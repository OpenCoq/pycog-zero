"""
Advanced Attention Mechanisms for Neural-Symbolic Integration

This module provides sophisticated attention mechanisms for enhanced neural-symbolic
processing in PyCog-Zero, including multi-scale, hierarchical, and cross-modal attention.
"""

# Try to import PyTorch components (graceful fallback if not installed)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available - advanced attention will use mock implementation")
    PYTORCH_AVAILABLE = False
    # Use mock from neural_symbolic_bridge
    from python.helpers.neural_symbolic_bridge import torch, MockModule

from typing import List, Dict, Tuple, Optional, Any
import math


class MultiScaleAttention:
    """Multi-scale attention mechanism for processing concepts at different granularities."""
    
    def __init__(self, embedding_dim: int, scales: List[int] = [1, 2, 4, 8]):
        """
        Initialize multi-scale attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            scales: List of attention scales (receptive field sizes)
        """
        self.embedding_dim = embedding_dim
        self.scales = scales
        
        if PYTORCH_AVAILABLE:
            # Create attention heads for each scale
            self.scale_attentions = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=max(1, embedding_dim // (64 * scale)),
                    batch_first=True
                ) for scale in scales
            ])
            
            # Scale fusion layer
            self.scale_fusion = torch.nn.Linear(embedding_dim * len(scales), embedding_dim)
            self.scale_weights = torch.nn.Parameter(torch.ones(len(scales)) / len(scales))
            
            # Layer normalization
            self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        else:
            self.scale_attentions = [MockModule() for _ in scales]
            self.scale_fusion = MockModule()
            self.scale_weights = torch.ones(len(scales)) / len(scales)
            self.layer_norm = MockModule()
    
    def forward(self, embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply multi-scale attention.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (attended_output, scale_attention_weights)
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        scale_outputs = []
        scale_weights = []
        
        # Apply attention at each scale
        for i, (scale, attention) in enumerate(zip(self.scales, self.scale_attentions)):
            # Create scale-specific view of embeddings
            if scale > 1 and embeddings.size(1) >= scale:
                # Group embeddings into scale-sized chunks for broader receptive field
                seq_len = embeddings.size(1)
                grouped_len = seq_len // scale
                if grouped_len > 0:
                    reshaped = embeddings[:, :grouped_len * scale, :].view(
                        embeddings.size(0), grouped_len, scale, self.embedding_dim
                    ).mean(dim=2)  # Average within each group
                else:
                    reshaped = embeddings
            else:
                reshaped = embeddings
            
            # Apply attention at this scale
            attended, weights = attention(reshaped, reshaped, reshaped, 
                                        attn_mask=attention_mask, need_weights=True)
            
            # Upsample back to original resolution if needed
            if scale > 1 and attended.size(1) != embeddings.size(1):
                attended = attended.repeat_interleave(scale, dim=1)[:, :embeddings.size(1), :]
            
            scale_outputs.append(attended)
            scale_weights.append(weights)
        
        # Fuse scale outputs with learned weights
        if len(scale_outputs) > 1:
            stacked_outputs = torch.stack(scale_outputs, dim=-1)  # [batch, seq, embed, scales]
            weighted_scales = stacked_outputs * self.scale_weights.view(1, 1, 1, -1)
            fused_output = weighted_scales.sum(dim=-1)
        else:
            fused_output = scale_outputs[0]
        
        # Apply layer normalization
        output = self.layer_norm(fused_output + embeddings)
        
        return output, scale_weights


class HierarchicalAttention:
    """Hierarchical attention for processing concept relationships at multiple levels."""
    
    def __init__(self, embedding_dim: int, num_levels: int = 3, reduction_factor: float = 0.5):
        """
        Initialize hierarchical attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_levels: Number of hierarchical levels
            reduction_factor: Factor by which to reduce sequence length at each level
        """
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.reduction_factor = reduction_factor
        
        if PYTORCH_AVAILABLE:
            # Attention mechanisms for each level
            self.level_attentions = torch.nn.ModuleList([
                torch.nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=max(1, embedding_dim // 64),
                    batch_first=True
                ) for _ in range(num_levels)
            ])
            
            # Pooling layers to create hierarchical representations
            self.pooling_layers = torch.nn.ModuleList([
                torch.nn.AdaptiveAvgPool1d(
                    output_size=max(1, int(100 * (reduction_factor ** (i + 1))))
                ) for i in range(num_levels - 1)
            ])
            
            # Cross-level attention for integrating hierarchy
            self.cross_level_attention = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=max(1, embedding_dim // 64),
                batch_first=True
            )
            
            self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        else:
            self.level_attentions = [MockModule() for _ in range(num_levels)]
            self.pooling_layers = [MockModule() for _ in range(num_levels - 1)]
            self.cross_level_attention = MockModule()
            self.layer_norm = MockModule()
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply hierarchical attention.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Tuple of (hierarchical_output, level_attention_weights)
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        level_representations = []
        level_weights = []
        current_repr = embeddings
        
        # Process each hierarchical level
        for level in range(self.num_levels):
            # Apply self-attention at current level
            attended, weights = self.level_attentions[level](
                current_repr, current_repr, current_repr, need_weights=True
            )
            
            level_representations.append(attended)
            level_weights.append(weights)
            
            # Pool for next level (except for the last level)
            if level < self.num_levels - 1:
                # Transpose for pooling: [batch, seq, embed] -> [batch, embed, seq]
                pooled = self.pooling_layers[level](attended.transpose(1, 2))
                current_repr = pooled.transpose(1, 2)  # Back to [batch, seq, embed]
        
        # Integrate across hierarchical levels using cross-level attention
        if len(level_representations) > 1:
            # Combine all levels (upsample lower levels to match the first level size)
            target_len = level_representations[0].size(1)
            upsampled_levels = []
            
            for i, level_repr in enumerate(level_representations):
                if level_repr.size(1) != target_len:
                    # Simple upsampling by repetition
                    repeat_factor = target_len // level_repr.size(1)
                    remainder = target_len % level_repr.size(1)
                    upsampled = level_repr.repeat_interleave(repeat_factor, dim=1)
                    if remainder > 0:
                        upsampled = torch.cat([upsampled, level_repr[:, :remainder, :]], dim=1)
                    upsampled_levels.append(upsampled)
                else:
                    upsampled_levels.append(level_repr)
            
            # Stack and apply cross-level attention
            stacked_levels = torch.stack(upsampled_levels, dim=1)  # [batch, levels, seq, embed]
            batch_size, num_levels, seq_len, embed_dim = stacked_levels.shape
            
            # Reshape for cross-level attention
            query = stacked_levels[:, 0, :, :].contiguous()  # Use first level as query
            key_value = stacked_levels.view(batch_size, num_levels * seq_len, embed_dim)
            
            cross_attended, _ = self.cross_level_attention(query, key_value, key_value, need_weights=False)
            output = self.layer_norm(cross_attended + embeddings)
        else:
            output = self.layer_norm(level_representations[0] + embeddings)
        
        return output, level_weights


class CrossModalAttention:
    """Cross-modal attention for neural-symbolic integration."""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8):
        """
        Initialize cross-modal attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        if PYTORCH_AVAILABLE:
            # Neural-to-symbolic attention
            self.neural_to_symbolic = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
            # Symbolic-to-neural attention
            self.symbolic_to_neural = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
            # Cross-modal fusion
            self.fusion_layer = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim * 2, embedding_dim * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim * 2, embedding_dim),
                torch.nn.LayerNorm(embedding_dim)
            )
            
            # Modal projection layers
            self.neural_projection = torch.nn.Linear(embedding_dim, embedding_dim)
            self.symbolic_projection = torch.nn.Linear(embedding_dim, embedding_dim)
            
            self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        else:
            self.neural_to_symbolic = MockModule()
            self.symbolic_to_neural = MockModule()
            self.fusion_layer = MockModule()
            self.neural_projection = MockModule()
            self.symbolic_projection = MockModule()
            self.layer_norm = MockModule()
    
    def forward(self, neural_embeddings: torch.Tensor, symbolic_embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply cross-modal attention between neural and symbolic representations.
        
        Args:
            neural_embeddings: Neural embeddings [batch_size, seq_len, embedding_dim]
            symbolic_embeddings: Symbolic embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Tuple of (fused_embeddings, attention_weights_dict)
        """
        if neural_embeddings.dim() == 2:
            neural_embeddings = neural_embeddings.unsqueeze(0)
        if symbolic_embeddings.dim() == 2:
            symbolic_embeddings = symbolic_embeddings.unsqueeze(0)
        
        # Project embeddings to cross-modal space
        neural_proj = self.neural_projection(neural_embeddings)
        symbolic_proj = self.symbolic_projection(symbolic_embeddings)
        
        # Neural-to-symbolic attention
        neural_attended, neural_weights = self.neural_to_symbolic(
            query=neural_proj,
            key=symbolic_proj,
            value=symbolic_proj,
            need_weights=True
        )
        
        # Symbolic-to-neural attention
        symbolic_attended, symbolic_weights = self.symbolic_to_neural(
            query=symbolic_proj,
            key=neural_proj,
            value=neural_proj,
            need_weights=True
        )
        
        # Cross-modal fusion
        concatenated = torch.cat([neural_attended, symbolic_attended], dim=-1)
        fused = self.fusion_layer(concatenated)
        
        # Residual connection with original neural embeddings
        output = self.layer_norm(fused + neural_embeddings)
        
        attention_weights = {
            'neural_to_symbolic': neural_weights,
            'symbolic_to_neural': symbolic_weights
        }
        
        return output, attention_weights


class TemporalAttention:
    """Temporal attention mechanism for sequential neural-symbolic reasoning."""
    
    def __init__(self, embedding_dim: int, memory_size: int = 100, num_heads: int = 8):
        """
        Initialize temporal attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            memory_size: Size of attention memory
            num_heads: Number of attention heads
        """
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Initialize memory buffer
        self.attention_memory = []
        self.memory_timestamps = []
        
        if PYTORCH_AVAILABLE:
            # Temporal self-attention
            self.temporal_attention = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
            # Memory attention
            self.memory_attention = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                batch_first=True
            )
            
            # Temporal fusion
            self.temporal_fusion = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim * 2, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim, embedding_dim),
                torch.nn.LayerNorm(embedding_dim)
            )
            
            # Position encoding
            self.positional_encoding = self._create_positional_encoding(embedding_dim)
            
            self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        else:
            self.temporal_attention = MockModule()
            self.memory_attention = MockModule()
            self.temporal_fusion = MockModule()
            self.positional_encoding = torch.zeros((1000, embedding_dim))
            self.layer_norm = MockModule()
    
    def _create_positional_encoding(self, d_model: int, max_len: int = 1000) -> torch.Tensor:
        """Create positional encoding for temporal sequences."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, embeddings: torch.Tensor, timestamp: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention with memory.
        
        Args:
            embeddings: Current embeddings [batch_size, seq_len, embedding_dim]
            timestamp: Current timestamp (optional)
            
        Returns:
            Tuple of (temporal_output, memory_attention_weights)
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(0):
            pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            embeddings_with_pos = embeddings + pos_encoding
        else:
            embeddings_with_pos = embeddings
        
        # Apply temporal self-attention
        temporal_attended, temporal_weights = self.temporal_attention(
            embeddings_with_pos, embeddings_with_pos, embeddings_with_pos, need_weights=True
        )
        
        # Memory-based attention if memory exists
        if self.attention_memory:
            memory_tensor = torch.stack(self.attention_memory[:self.memory_size])
            if memory_tensor.dim() == 2:
                memory_tensor = memory_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Attend to memory
            memory_attended, memory_weights = self.memory_attention(
                query=temporal_attended,
                key=memory_tensor,
                value=memory_tensor,
                need_weights=True
            )
            
            # Fuse temporal and memory attention
            concatenated = torch.cat([temporal_attended, memory_attended], dim=-1)
            fused = self.temporal_fusion(concatenated)
        else:
            fused = temporal_attended
            memory_weights = torch.zeros((batch_size, seq_len, 1))
        
        # Update memory
        self._update_memory(temporal_attended.mean(dim=1), timestamp)  # Store averaged representation
        
        # Final output with residual connection
        output = self.layer_norm(fused + embeddings)
        
        return output, memory_weights
    
    def _update_memory(self, new_embedding: torch.Tensor, timestamp: Optional[int] = None):
        """Update attention memory with new embedding."""
        if timestamp is None:
            timestamp = len(self.attention_memory)
        
        # Add to memory
        self.attention_memory.append(new_embedding.detach())
        self.memory_timestamps.append(timestamp)
        
        # Maintain memory size
        if len(self.attention_memory) > self.memory_size:
            self.attention_memory.pop(0)
            self.memory_timestamps.pop(0)
    
    def clear_memory(self):
        """Clear attention memory."""
        self.attention_memory.clear()
        self.memory_timestamps.clear()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of attention memory state."""
        return {
            'memory_size': len(self.attention_memory),
            'memory_capacity': self.memory_size,
            'timestamps': self.memory_timestamps.copy() if self.memory_timestamps else [],
            'memory_utilization': len(self.attention_memory) / self.memory_size if self.memory_size > 0 else 0
        }


class MetaAttention:
    """Meta-attention mechanism for attention over attention patterns."""
    
    def __init__(self, embedding_dim: int, num_attention_types: int = 4):
        """
        Initialize meta-attention.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_attention_types: Number of different attention mechanisms to meta-attend over
        """
        self.embedding_dim = embedding_dim
        self.num_attention_types = num_attention_types
        
        if PYTORCH_AVAILABLE:
            # Meta-attention over attention patterns
            self.meta_attention = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=max(1, embedding_dim // 64),
                batch_first=True
            )
            
            # Attention type embeddings
            self.attention_type_embeddings = torch.nn.Embedding(num_attention_types, embedding_dim)
            
            # Attention weight fusion
            self.weight_fusion = torch.nn.Sequential(
                torch.nn.Linear(num_attention_types, num_attention_types * 2),
                torch.nn.ReLU(),
                torch.nn.Linear(num_attention_types * 2, num_attention_types),
                torch.nn.Softmax(dim=-1)
            )
            
            self.layer_norm = torch.nn.LayerNorm(embedding_dim)
        else:
            self.meta_attention = MockModule()
            self.attention_type_embeddings = MockModule()
            self.weight_fusion = MockModule()
            self.layer_norm = MockModule()
    
    def forward(self, embeddings: torch.Tensor, attention_weights_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply meta-attention over multiple attention mechanisms.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            attention_weights_list: List of attention weight tensors from different mechanisms
            
        Returns:
            Tuple of (meta_attended_output, meta_attention_weights)
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Create attention type representations
        attention_representations = []
        for i, weights in enumerate(attention_weights_list):
            # Get type embedding
            type_embedding = self.attention_type_embeddings(torch.tensor(i % self.num_attention_types))
            type_embedding = type_embedding.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
            
            # Combine with attention pattern (using mean of weights as summary)
            if weights.dim() > 2:
                weight_summary = weights.mean(dim=list(range(1, weights.dim())))
            else:
                weight_summary = weights.mean(dim=-1)
            
            # Create attention representation
            attention_repr = type_embedding + weight_summary.unsqueeze(-1) * 0.1
            attention_representations.append(attention_repr)
        
        if attention_representations:
            # Stack attention representations
            attention_stack = torch.cat(attention_representations, dim=1)
            
            # Apply meta-attention
            meta_attended, meta_weights = self.meta_attention(
                query=embeddings.mean(dim=1, keepdim=True),  # Use mean as query
                key=attention_stack,
                value=attention_stack,
                need_weights=True
            )
            
            # Broadcast back to sequence length
            meta_attended = meta_attended.expand(-1, seq_len, -1)
            
            # Compute fusion weights for different attention types
            attention_importance = torch.stack([w.mean() for w in attention_weights_list])
            fusion_weights = self.weight_fusion(attention_importance.unsqueeze(0))
            
            # Apply fusion weights to original embeddings
            output = self.layer_norm(embeddings + meta_attended * fusion_weights.mean())
        else:
            output = embeddings
            meta_weights = torch.zeros((batch_size, 1, 1))
        
        return output, meta_weights