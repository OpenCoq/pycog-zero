"""
ECAN (Economic Attention Networks) Coordinator for PyCog-Zero

This module provides centralized coordination of attention allocation across
all cognitive tools in the PyCog-Zero framework, ensuring consistent and
efficient attention management.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types, Atom
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False
    # Fallback type definitions for type hints
    AtomSpace = Any
    Atom = Any

# Try to import ECAN for attention allocation
try:
    from opencog.ecan import AttentionBank, ECANAgent
    ECAN_AVAILABLE = True
except ImportError:
    ECAN_AVAILABLE = False

# Try to import PyTorch for neural attention integration
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


@dataclass
class AttentionRequest:
    """Represents a request for attention allocation from a cognitive tool."""
    tool_name: str
    priority: float
    context: str
    concepts: List[str]
    importance_multiplier: float = 1.0
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class AttentionAllocation:
    """Represents the result of attention allocation."""
    total_sti_allocated: int
    concept_allocations: Dict[str, int]
    tool_allocations: Dict[str, int]
    attention_entropy: float
    allocation_timestamp: float


class ECANCoordinator:
    """
    Centralized ECAN coordinator for managing attention across cognitive tools.
    
    This class coordinates attention allocation between meta_cognition, 
    cognitive_reasoning, cognitive_memory, and other tools that require
    attention management.
    """
    
    def __init__(self, shared_atomspace: Optional[AtomSpace] = None):
        """Initialize the ECAN coordinator."""
        self.atomspace = shared_atomspace
        self.attention_bank = None
        self.ecan = None
        self.initialized = False
        
        # Cross-tool tracking
        self.active_tools = set()
        self.attention_requests = {}
        self.allocation_history = []
        self.total_sti_budget = 1000  # Default STI budget
        
        # Fallback mechanisms
        self.fallback_weights = {}
        self.fallback_priorities = {}
        
        # Performance metrics
        self.metrics = {
            "total_allocations": 0,
            "average_entropy": 0.0,
            "cross_tool_interactions": 0,
            "attention_conflicts": 0
        }
        
        self._initialize_ecan_components()
    
    def _initialize_ecan_components(self):
        """Initialize ECAN components if available."""
        if OPENCOG_AVAILABLE and self.atomspace:
            try:
                # Initialize ECAN components
                if ECAN_AVAILABLE:
                    self.attention_bank = AttentionBank(self.atomspace)
                    self.ecan = ECANAgent(self.atomspace)
                    self.initialized = True
                    print("✓ ECAN Coordinator initialized with OpenCog components")
                else:
                    print("⚠️ ECAN not available - coordinator using fallback mechanisms")
            except Exception as e:
                print(f"⚠️ ECAN initialization failed: {e}")
                print("ℹ️ ECAN Coordinator using fallback mechanisms")
        else:
            print("⚠️ OpenCog not available - ECAN Coordinator using fallback mechanisms")
    
    def register_tool(self, tool_name: str, default_priority: float = 1.0):
        """Register a cognitive tool with the ECAN coordinator."""
        self.active_tools.add(tool_name)
        self.fallback_priorities[tool_name] = default_priority
        print(f"✓ Registered cognitive tool '{tool_name}' with ECAN coordinator")
    
    def unregister_tool(self, tool_name: str):
        """Unregister a cognitive tool from the ECAN coordinator."""
        self.active_tools.discard(tool_name)
        self.attention_requests.pop(tool_name, None)
        self.fallback_priorities.pop(tool_name, None)
        print(f"ℹ️ Unregistered cognitive tool '{tool_name}' from ECAN coordinator")
    
    def request_attention(self, request: AttentionRequest) -> bool:
        """
        Request attention allocation for a cognitive tool.
        
        Args:
            request: AttentionRequest object containing allocation requirements
            
        Returns:
            bool: True if request was accepted, False otherwise
        """
        # Validate request
        if request.tool_name not in self.active_tools:
            print(f"⚠️ Tool '{request.tool_name}' not registered with ECAN coordinator")
            return False
        
        # Store the attention request
        self.attention_requests[request.tool_name] = request
        
        # If we have real ECAN, process immediately
        if self.initialized:
            return self._process_ecan_request(request)
        else:
            return self._process_fallback_request(request)
    
    def _process_ecan_request(self, request: AttentionRequest) -> bool:
        """Process attention request using real ECAN components."""
        try:
            # Create concept nodes for the request
            concept_nodes = []
            for concept in request.concepts:
                node_name = f"{request.tool_name}_{concept}"
                node = self.atomspace.add_node(types.ConceptNode, node_name)
                concept_nodes.append(node)
            
            # Calculate STI allocation based on priority
            base_sti = int(request.priority * request.importance_multiplier * 100)
            
            # Distribute STI across concepts
            sti_per_concept = max(10, base_sti // len(request.concepts))
            
            for node in concept_nodes:
                self.attention_bank.set_sti(node, sti_per_concept)
            
            # Run ECAN dynamics
            self.ecan.run_cycle()
            
            print(f"✓ ECAN attention allocated for {request.tool_name}: {len(concept_nodes)} concepts, {base_sti} total STI")
            return True
            
        except Exception as e:
            print(f"⚠️ ECAN processing error for {request.tool_name}: {e}")
            return self._process_fallback_request(request)
    
    def _process_fallback_request(self, request: AttentionRequest) -> bool:
        """Process attention request using fallback mechanisms."""
        # Simple priority-based weighting
        weight = request.priority * request.importance_multiplier
        normalized_weight = min(1.0, weight / 10.0)  # Normalize to 0-1
        
        # Store fallback weights for the tool
        self.fallback_weights[request.tool_name] = {
            "weight": normalized_weight,
            "concepts": request.concepts,
            "timestamp": request.timestamp
        }
        
        print(f"✓ Fallback attention allocated for {request.tool_name}: weight={normalized_weight:.3f}")
        return True
    
    def get_attention_allocation(self) -> AttentionAllocation:
        """Get current attention allocation across all tools."""
        if self.initialized:
            return self._get_ecan_allocation()
        else:
            return self._get_fallback_allocation()
    
    def _get_ecan_allocation(self) -> AttentionAllocation:
        """Get attention allocation using ECAN components."""
        try:
            # Get all concept nodes
            concept_nodes = self.atomspace.get_atoms_by_type(types.ConceptNode)
            
            concept_allocations = {}
            tool_allocations = {}
            total_sti = 0
            
            for node in concept_nodes:
                try:
                    sti = self.attention_bank.get_sti(node)
                    if sti > 0:
                        node_name = str(node.name) if hasattr(node, 'name') else str(node)
                        concept_allocations[node_name] = sti
                        total_sti += sti
                        
                        # Extract tool name from node name
                        if '_' in node_name:
                            tool_name = node_name.split('_')[0]
                            tool_allocations[tool_name] = tool_allocations.get(tool_name, 0) + sti
                except:
                    continue
            
            # Calculate attention entropy
            if total_sti > 0:
                entropy = self._calculate_attention_entropy(concept_allocations, total_sti)
            else:
                entropy = 0.0
            
            allocation = AttentionAllocation(
                total_sti_allocated=total_sti,
                concept_allocations=concept_allocations,
                tool_allocations=tool_allocations,
                attention_entropy=entropy,
                allocation_timestamp=time.time()
            )
            
            # Update metrics
            self.metrics["total_allocations"] += 1
            self.metrics["average_entropy"] = (
                (self.metrics["average_entropy"] * (self.metrics["total_allocations"] - 1) + entropy) /
                self.metrics["total_allocations"]
            )
            
            return allocation
            
        except Exception as e:
            print(f"⚠️ Error getting ECAN allocation: {e}")
            return self._get_fallback_allocation()
    
    def _get_fallback_allocation(self) -> AttentionAllocation:
        """Get attention allocation using fallback mechanisms."""
        concept_allocations = {}
        tool_allocations = {}
        total_weight = 0
        
        for tool_name, weight_info in self.fallback_weights.items():
            weight = weight_info["weight"]
            concepts = weight_info["concepts"]
            
            tool_allocations[tool_name] = int(weight * 100)
            total_weight += weight
            
            for concept in concepts:
                concept_key = f"{tool_name}_{concept}"
                concept_allocations[concept_key] = int(weight * 100 / len(concepts))
        
        # Calculate entropy
        if total_weight > 0:
            entropy = self._calculate_attention_entropy(
                {k: v/100.0 for k, v in concept_allocations.items()}, 
                total_weight
            )
        else:
            entropy = 0.0
        
        return AttentionAllocation(
            total_sti_allocated=int(total_weight * 100),
            concept_allocations=concept_allocations,
            tool_allocations=tool_allocations,
            attention_entropy=entropy,
            allocation_timestamp=time.time()
        )
    
    def _calculate_attention_entropy(self, allocations: Dict[str, int], total: int) -> float:
        """Calculate attention entropy (measure of attention distribution)."""
        if total == 0:
            return 0.0
        
        import math
        entropy = 0.0
        
        for allocation in allocations.values():
            if allocation > 0:
                p = allocation / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def get_tool_priority(self, tool_name: str) -> float:
        """Get current attention priority for a tool."""
        if tool_name in self.attention_requests:
            return self.attention_requests[tool_name].priority
        return self.fallback_priorities.get(tool_name, 1.0)
    
    def synchronize_attention(self) -> Dict[str, Any]:
        """Synchronize attention across all registered tools."""
        allocation = self.get_attention_allocation()
        
        # Create synchronization data
        sync_data = {
            "timestamp": time.time(),
            "active_tools": list(self.active_tools),
            "allocation": allocation,
            "metrics": self.metrics.copy(),
            "ecan_available": self.initialized
        }
        
        # Store in history (limited)
        self.allocation_history.append(sync_data)
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]
        
        self.metrics["cross_tool_interactions"] += 1
        
        return sync_data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ECAN coordinator performance metrics."""
        return {
            "ecan_available": self.initialized,
            "active_tools": len(self.active_tools),
            "total_requests": len(self.attention_requests),
            "allocation_history_size": len(self.allocation_history),
            **self.metrics
        }


# Singleton instance for cross-tool coordination
_ecan_coordinator = None


def get_ecan_coordinator(shared_atomspace: Optional[AtomSpace] = None) -> ECANCoordinator:
    """Get or create the global ECAN coordinator instance."""
    global _ecan_coordinator
    
    if _ecan_coordinator is None:
        _ecan_coordinator = ECANCoordinator(shared_atomspace)
    
    return _ecan_coordinator


def register_tool_with_ecan(tool_name: str, default_priority: float = 1.0):
    """Convenience function to register a tool with the global ECAN coordinator."""
    coordinator = get_ecan_coordinator()
    coordinator.register_tool(tool_name, default_priority)


def request_attention_for_tool(tool_name: str, priority: float, context: str, 
                             concepts: List[str], importance_multiplier: float = 1.0) -> bool:
    """Convenience function to request attention for a tool."""
    coordinator = get_ecan_coordinator()
    request = AttentionRequest(
        tool_name=tool_name,
        priority=priority,
        context=context,
        concepts=concepts,
        importance_multiplier=importance_multiplier
    )
    return coordinator.request_attention(request)