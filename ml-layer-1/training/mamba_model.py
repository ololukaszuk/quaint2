"""
Mamba Model Module

Implements a simplified Mamba (State Space Model) for sequence modeling.
This implementation is TorchScript-compatible for export.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import MambaConfig, NUM_FEATURES, SEQUENCE_LENGTH, NUM_HORIZONS


class MambaBlock(nn.Module):
    """
    Single Mamba block implementing selective state space model.
    
    Simplified for TorchScript compatibility.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A parameter (diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        # Convolution (transpose for conv1d)
        x_conv = x_in.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        
        # Apply SiLU activation
        x_conv = F.silu(x_conv)
        
        # SSM computation (simplified)
        A = -torch.exp(self.A_log)  # (d_state,)
        
        # Project to get B and C
        x_ssm = self.x_proj(x_conv)
        B, C = x_ssm.chunk(2, dim=-1)  # (batch, seq_len, d_state)
        
        # Delta (dt) projection
        dt = F.softplus(self.dt_proj(x_conv))  # (batch, seq_len, d_inner)
        
        # Discretized SSM (simplified scan)
        y = self._ssm_scan(x_conv, dt, A, B, C)
        
        # Add skip connection with D
        y = y + x_conv * self.D
        
        # Gate and output
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        return output
    
    def _ssm_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified selective scan (sequential, TorchScript compatible).
        
        Args:
            u: Input (batch, seq_len, d_inner)
            delta: Time step (batch, seq_len, d_inner)
            A: State matrix diagonal (d_state,)
            B: Input-to-state (batch, seq_len, d_state)
            C: State-to-output (batch, seq_len, d_state)
            
        Returns:
            Output (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[0]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Get current inputs
            u_t = u[:, t, :]  # (batch, d_inner)
            delta_t = delta[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :]  # (batch, d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            
            # Discretize A and B
            dA = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (batch, d_inner, d_state)
            dB = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_inner, d_state)
            
            # State update: h = dA * h + dB * u
            h = dA * h + dB * u_t.unsqueeze(-1)
            
            # Output: y = C * h
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class MambaModel(nn.Module):
    """
    Complete Mamba model for time series prediction.
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        
        self.config = config
        
        # Input embedding
        self.input_proj = nn.Linear(config.input_size, config.d_model)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )
            for _ in range(config.n_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(config.d_model)
            for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.output_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Predictions (batch, output_size)
        """
        # Input projection
        h = self.input_proj(x)
        h = self.dropout(h)
        
        # Mamba blocks with residual connections
        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h = norm(h)
            h = layer(h)
            h = self.dropout(h)
            h = h + residual
        
        # Final norm
        h = self.final_norm(h)
        
        # Take last timestep
        h = h[:, -1, :]
        
        # Output projection
        output = self.output_proj(h)
        
        return output


def create_mamba_model(config: Optional[MambaConfig] = None) -> MambaModel:
    """
    Create a new Mamba model.
    
    Args:
        config: Model configuration (uses default if None)
        
    Returns:
        Initialized MambaModel
    """
    if config is None:
        config = MambaConfig()
    return MambaModel(config)


def load_mamba_model(path: str, config: Optional[MambaConfig] = None) -> MambaModel:
    """
    Load a trained Mamba model from checkpoint.
    
    Args:
        path: Path to checkpoint file
        config: Model configuration
        
    Returns:
        Loaded MambaModel
    """
    if config is None:
        config = MambaConfig()
    
    model = MambaModel(config)
    checkpoint = torch.load(path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def export_mamba_torchscript(model: MambaModel, output_path: str) -> None:
    """
    Export Mamba model to TorchScript format using tracing.
    
    Args:
        model: Trained MambaModel
        output_path: Path for output .pt file
    """
    model.eval()
    
    # Create example input for tracing
    example_input = torch.randn(1, SEQUENCE_LENGTH, NUM_FEATURES)
    
    # Use trace instead of script to avoid Self type annotation issues
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    
    # Verify output
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = traced(example_input)
    
    if not torch.allclose(original_output, traced_output, atol=1e-5):
        raise RuntimeError("TorchScript traced output doesn't match original")
    
    # Save
    traced.save(output_path)
    print(f"Exported TorchScript model to {output_path}")