"""
Neural-Symbolic Bridge for PyTorch-OpenCog Integration

This module provides a bridge between PyTorch neural networks and OpenCog symbolic
reasoning for the PyCog-Zero cognitive architecture. It enables bidirectional 
conversion between neural tensor representations and symbolic AtomSpace atoms.
"""

# Try to import PyTorch components (graceful fallback if not installed)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch not available - neural-symbolic bridge will use mock implementation")
    PYTORCH_AVAILABLE = False
    # Mock PyTorch components for fallback
    class MockTensor:
        def __init__(self, data=None, shape=None):
            self.shape = shape or (0,)
            self._data = data
            
        def unsqueeze(self, dim):
            return MockTensor(shape=tuple(list(self.shape[:dim]) + [1] + list(self.shape[dim:])))
        
        def squeeze(self, dim=None):
            return MockTensor(shape=tuple(s for i, s in enumerate(self.shape) if i != dim or s != 1))
        
        def detach(self):
            return self
        
        def tolist(self):
            return []
        
        def mean(self, dim=None):
            return MockTensor()
        
        def size(self, dim=None):
            return self.shape[dim] if dim is not None else self.shape
        
        def norm(self, dim=None):
            return MockTensor()
        
        def sum(self, dim=None):
            return MockTensor()
        
        def argmax(self, dim=None):
            return 0
        
        def __getitem__(self, idx):
            return MockTensor()
        
        def __len__(self):
            return self.shape[0] if self.shape else 0
        
        def __mul__(self, other):
            return MockTensor(shape=self.shape)
        
        def __add__(self, other):
            return MockTensor(shape=self.shape)
        
        def __sub__(self, other):
            return MockTensor(shape=self.shape)
        
        def __truediv__(self, other):
            return MockTensor(shape=self.shape)
        
        def __iter__(self):
            for i in range(len(self)):
                yield MockTensor()
        
        def __float__(self):
            return 0.0
        
        def item(self):
            return 0.0
    
    class MockModule:
        def __init__(self, *args, **kwargs):
            pass
        def parameters(self):
            return []
        def train(self):
            pass
        def eval(self):
            pass
    
    class MockOptim:
        def __init__(self, *args, **kwargs):
            pass
        def zero_grad(self):
            pass
        def step(self):
            pass
    
    class MockNN:
        Sequential = MockModule
        Linear = MockModule
        ReLU = MockModule
        Tanh = MockModule
        MultiheadAttention = MockModule
        LayerNorm = MockModule
        MSELoss = MockModule
        
        class init:
            @staticmethod
            def xavier_uniform_(tensor):
                pass
            @staticmethod
            def zeros_(tensor):
                pass
    
    # Create mock torch module
    class MockTorch:
        Tensor = MockTensor
        tensor = lambda data, dtype=None: MockTensor(shape=(len(data),) if hasattr(data, '__len__') else (1,))
        randn = lambda *args, **kwargs: MockTensor(shape=args)
        stack = lambda tensors, dim=0: MockTensor(shape=(len(tensors), *tensors[0].shape) if tensors and hasattr(tensors[0], 'shape') else (len(tensors),))
        empty = lambda shape: MockTensor(shape=shape)
        zeros = lambda shape: MockTensor(shape=shape)
        cat = lambda tensors, dim=0: MockTensor()
        norm = lambda tensor, dim=None: MockTensor()
        sum = lambda tensor, dim=None: MockTensor()
        log = lambda tensor: MockTensor()
        argmax = lambda tensor, dim=None: 0
        equal = lambda a, b: True
        allclose = lambda a, b, **kwargs: True
        ones_like = lambda tensor: MockTensor(shape=tensor.shape)
        isnan = lambda tensor: MockTensor()
        isinf = lambda tensor: MockTensor()
        
        class optim:
            Adam = MockOptim
    
    torch = MockTorch()
    torch.nn = MockNN()

# Try to import numpy (graceful fallback if not installed)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️ NumPy not available - using basic Python fallback")
    NUMPY_AVAILABLE = False
    # Mock numpy for basic functionality
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod 
        def random(shape):
            return [0.1] * shape if isinstance(shape, int) else [[0.1] * shape[1]] * shape[0]
    
    np = MockNumpy()

from typing import List, Dict, Tuple, Optional, Any
import hashlib

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("⚠️ OpenCog not available - neural-symbolic bridge will use mock implementation")
    OPENCOG_AVAILABLE = False
    # Mock AtomSpace and types for fallback
    class MockAtomSpace:
        def add_node(self, atom_type, name):
            return MockAtom(name)
    
    class MockAtom:
        def __init__(self, name):
            self.name = name
            self.tv = None
    
    class MockTypes:
        ConceptNode = "ConceptNode"
        PredicateNode = "PredicateNode"
        NumberNode = "NumberNode"
    
    types = MockTypes()


class NeuralSymbolicBridge:
    """Bridge between PyTorch neural networks and OpenCog symbolic reasoning."""
    
    def __init__(self, atomspace: Any = None, embedding_dim: int = 128):
        """
        Initialize the neural-symbolic bridge.
        
        Args:
            atomspace: OpenCog AtomSpace instance (optional, will create if needed)
            embedding_dim: Dimension of neural embeddings
        """
        self.embedding_dim = embedding_dim
        self.atom_embeddings = {}  # Cache for atom -> embedding mappings
        
        # Initialize AtomSpace
        if OPENCOG_AVAILABLE:
            self.atomspace = atomspace if atomspace is not None else AtomSpace()
            try:
                initialize_opencog(self.atomspace)
                self.opencog_initialized = True
            except Exception as e:
                print(f"⚠️ OpenCog initialization warning: {e}")
                self.opencog_initialized = False
        else:
            self.atomspace = MockAtomSpace()
            self.opencog_initialized = False
        
        # Neural network for atom embedding learning
        if PYTORCH_AVAILABLE:
            self.embedding_network = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, embedding_dim),
                torch.nn.Tanh()
            )
        else:
            self.embedding_network = MockModule()
        
        # Initialize embedding network parameters
        self._initialize_embedding_network()
    
    def _initialize_embedding_network(self):
        """Initialize the embedding network with proper weights."""
        if not PYTORCH_AVAILABLE:
            return
            
        for layer in self.embedding_network:
            if isinstance(layer, torch.nn.Linear):
                # Xavier initialization for linear layers
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def atomspace_to_tensor(self, atoms: List[Any]) -> torch.Tensor:
        """
        Convert AtomSpace atoms to tensor representation for neural processing.
        
        Args:
            atoms: List of OpenCog atoms
            
        Returns:
            torch.Tensor: Tensor representation of atoms [num_atoms, embedding_dim]
        """
        if not atoms:
            return torch.empty((0, self.embedding_dim))
        
        embeddings = []
        for atom in atoms:
            if atom not in self.atom_embeddings:
                # Create initial embedding based on atom type and content
                atom_vector = self.create_atom_embedding(atom)
                self.atom_embeddings[atom] = atom_vector
            
            embeddings.append(self.atom_embeddings[atom])
        
        return torch.stack(embeddings)
    
    def tensor_to_atomspace(self, tensor: torch.Tensor, atom_types: List[str] = None, 
                          atom_names: List[str] = None) -> List[Any]:
        """
        Convert tensor representations back to AtomSpace atoms.
        
        Args:
            tensor: Tensor of embeddings [num_atoms, embedding_dim]
            atom_types: Optional list of atom type strings
            atom_names: Optional list of atom names
            
        Returns:
            List of OpenCog atoms
        """
        atoms = []
        for i, embedding in enumerate(tensor):
            # Determine atom type and name
            if OPENCOG_AVAILABLE:
                atom_type = getattr(types, atom_types[i]) if atom_types and i < len(atom_types) else types.ConceptNode
            else:
                atom_type = atom_types[i] if atom_types and i < len(atom_types) else "ConceptNode"
            
            atom_name = atom_names[i] if atom_names and i < len(atom_names) else f"neural_concept_{i}"
            
            # Create or find atom in AtomSpace
            atom = self.atomspace.add_node(atom_type, atom_name)
            
            # Cache the embedding
            self.atom_embeddings[atom] = embedding.detach()
            atoms.append(atom)
        
        return atoms
    
    def create_atom_embedding(self, atom: Any) -> torch.Tensor:
        """
        Create embedding for OpenCog atom based on its properties.
        
        Args:
            atom: OpenCog atom instance
            
        Returns:
            torch.Tensor: Embedding vector for the atom
        """
        # Start with a base embedding based on atom type
        type_embedding = torch.randn(self.embedding_dim) * 0.1
        
        # Modify based on atom name/content if available
        if hasattr(atom, 'name') and atom.name:
            # Use deterministic hash for consistent embeddings
            name_hash = int(hashlib.md5(atom.name.encode()).hexdigest()[:8], 16) % 1000000
            name_component = torch.tensor([name_hash / 1000000.0] * self.embedding_dim)
            type_embedding = type_embedding + name_component * 0.1
        
        # Add truth value information if available (OpenCog specific)
        if hasattr(atom, 'tv') and atom.tv:
            try:
                confidence = getattr(atom.tv, 'confidence', 0.5)
                strength = getattr(atom.tv, 'mean', 0.5)
                tv_size = min(self.embedding_dim // 2, 64)  # Prevent overflow
                tv_embedding = torch.tensor([strength, confidence] * tv_size)[:self.embedding_dim]
                # Pad if necessary
                if len(tv_embedding) < self.embedding_dim:
                    padding = torch.zeros(self.embedding_dim - len(tv_embedding))
                    tv_embedding = torch.cat([tv_embedding, padding])
                type_embedding = type_embedding + tv_embedding * 0.1
            except (AttributeError, TypeError):
                pass  # Skip if truth value format is unexpected
        
        return type_embedding
    
    def update_embedding(self, atom: Any, embedding: torch.Tensor):
        """
        Update the cached embedding for an atom.
        
        Args:
            atom: OpenCog atom
            embedding: New embedding vector
        """
        self.atom_embeddings[atom] = embedding.detach()
    
    def get_embedding(self, atom: Any) -> Optional[torch.Tensor]:
        """
        Get the cached embedding for an atom.
        
        Args:
            atom: OpenCog atom
            
        Returns:
            Cached embedding or None if not found
        """
        return self.atom_embeddings.get(atom)
    
    def embed_concepts(self, concepts: List[str]) -> torch.Tensor:
        """
        Create embeddings for a list of concept strings.
        
        Args:
            concepts: List of concept names
            
        Returns:
            torch.Tensor: Embeddings for concepts [num_concepts, embedding_dim]
        """
        atoms = []
        for concept in concepts:
            if OPENCOG_AVAILABLE:
                atom = self.atomspace.add_node(types.ConceptNode, concept)
            else:
                atom = MockAtom(concept)
            atoms.append(atom)
        
        return self.atomspace_to_tensor(atoms)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.atom_embeddings.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self.atom_embeddings)
    
    def train_embeddings(self, atoms: List[Any], target_embeddings: torch.Tensor, 
                        learning_rate: float = 0.001, epochs: int = 100):
        """
        Train the embedding network to produce target embeddings for given atoms.
        
        Args:
            atoms: List of atoms to train on
            target_embeddings: Target embeddings [num_atoms, embedding_dim]
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        if not PYTORCH_AVAILABLE:
            print("⚠️ PyTorch not available - training skipped")
            return
            
        if not atoms or target_embeddings.size(0) != len(atoms):
            raise ValueError("Atoms and target_embeddings must have matching lengths")
        
        optimizer = torch.optim.Adam(self.embedding_network.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Get initial embeddings
        initial_embeddings = self.atomspace_to_tensor(atoms)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass through embedding network
            refined_embeddings = self.embedding_network(initial_embeddings)
            
            # Compute loss
            loss = criterion(refined_embeddings, target_embeddings)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update cached embeddings
            for i, atom in enumerate(atoms):
                self.atom_embeddings[atom] = refined_embeddings[i].detach()
            
            if epoch % 20 == 0:
                print(f"Training epoch {epoch}, loss: {loss.item():.4f}")


class CognitiveAttentionMechanism:
    """Attention mechanism integrating OpenCog ECAN with neural networks."""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8):
        """
        Initialize the cognitive attention mechanism.
        
        Args:
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
        super().__init__() if PYTORCH_AVAILABLE else None
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head attention mechanism
        if PYTORCH_AVAILABLE:
            self.attention = torch.nn.MultiheadAttention(
                embed_dim=embedding_dim, 
                num_heads=num_heads, 
                batch_first=True
            )
            
            # Layer norm for stability
            self.layer_norm = torch.nn.LayerNorm(embedding_dim)
            
            # Feedforward network for post-attention processing
            self.feedforward = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim * 4),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim * 4, embedding_dim)
            )
        else:
            # Mock components for fallback
            self.attention = MockModule()
            self.layer_norm = MockModule()
            self.feedforward = MockModule()
        
    def forward(self, atom_embeddings: torch.Tensor, 
                ecan_weights: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism with ECAN weight integration.
        
        Args:
            atom_embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            ecan_weights: ECAN STI/LTI values [batch_size, seq_len] (optional)
            attention_mask: Attention mask [seq_len, seq_len] (optional)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        # Ensure proper dimensions
        if atom_embeddings.dim() == 2:
            atom_embeddings = atom_embeddings.unsqueeze(0)  # Add batch dimension
        
        # Apply ECAN weighting if provided
        if ecan_weights is not None:
            if ecan_weights.dim() == 1:
                ecan_weights = ecan_weights.unsqueeze(0)  # Add batch dimension
            # Scale embeddings by ECAN weights
            weighted_embeddings = atom_embeddings * ecan_weights.unsqueeze(-1)
        else:
            weighted_embeddings = atom_embeddings
        
        # Layer normalization
        normalized_embeddings = self.layer_norm(weighted_embeddings)
        
        # Apply multi-head attention
        attended_output, attention_weights = self.attention(
            query=normalized_embeddings,
            key=normalized_embeddings, 
            value=normalized_embeddings,
            attn_mask=attention_mask,
            need_weights=True
        )
        
        # Residual connection
        attended_output = attended_output + weighted_embeddings
        
        # Apply feedforward network with residual connection
        ff_output = self.feedforward(attended_output)
        final_output = self.layer_norm(ff_output + attended_output)
        
        return final_output, attention_weights
    
    def compute_ecan_weights(self, atoms: List[Any]) -> torch.Tensor:
        """
        Compute ECAN-style attention weights from atom importance values.
        
        Args:
            atoms: List of OpenCog atoms
            
        Returns:
            torch.Tensor: ECAN weights [num_atoms]
        """
        weights = []
        for atom in atoms:
            # Try to extract STI (Short-Term Importance) if available
            if hasattr(atom, 'sti'):
                sti_value = getattr(atom, 'sti', 0)
            elif hasattr(atom, 'av') and hasattr(atom.av, 'sti'):
                sti_value = atom.av.sti
            else:
                # Default importance value
                sti_value = 1.0
            
            weights.append(max(0.1, sti_value))  # Ensure positive weights
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        # Normalize to sum to 1
        return weights_tensor / weights_tensor.sum()