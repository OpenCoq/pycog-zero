"""
Enhanced Cognitive Memory Tool for Agent-Zero
Integrates Agent-Zero memory with OpenCog AtomSpace for persistent cognitive memory
Now with performance optimization for large-scale processing
"""

from python.helpers.tool import Tool, Response
from python.helpers.performance_optimizer import get_performance_optimizer, optimize
from python.helpers import files
import json
import pickle
import os
import asyncio
import time
from typing import Dict, List, Any, Optional

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - install with: pip install opencog-atomspace opencog-python")
    OPENCOG_AVAILABLE = False

# Import ECAN coordinator for cross-tool attention management
try:
    from python.helpers.ecan_coordinator import (
        get_ecan_coordinator, register_tool_with_ecan, 
        request_attention_for_tool, AttentionRequest
    )
    ECAN_COORDINATOR_AVAILABLE = True
except ImportError:
    ECAN_COORDINATOR_AVAILABLE = False


class CognitiveMemoryTool(Tool):
    """Integrate Agent-Zero memory with OpenCog AtomSpace."""
    
    def _initialize_if_needed(self):
        """Initialize the cognitive memory system if not already done."""
        if hasattr(self, '_cognitive_initialized'):
            return
        
        self._cognitive_initialized = True
        self.atomspace = None
        self.memory_file = files.get_abs_path("memory/cognitive_atomspace.pkl")
        self.initialized = False
        self._fallback_memory = {"atoms": [], "metadata": {"total_atoms": 0}}
        
        # Initialize performance optimizer
        optimizer_config = {
            'cache_size': 2000,  # Large cache for memory operations
            'batch_size': 100,   # Bigger batches for memory operations
            'batch_wait_time': 0.5,
            'memory_pool_size': 200
        }
        self.performance_optimizer = get_performance_optimizer(optimizer_config)
        
        # Performance monitoring
        self.memory_stats = {
            'total_operations': 0,
            'cached_operations': 0,
            'batch_operations': 0,
            'storage_operations': 0,
            'avg_response_time': 0.0
        }
        
        # Setup batch handlers
        self._setup_batch_handlers()
        
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.load_persistent_memory()
                self.initialized = True
                print("✓ OpenCog cognitive memory initialized")
                
                # Register with ECAN coordinator for attention management
                if ECAN_COORDINATOR_AVAILABLE:
                    register_tool_with_ecan("cognitive_memory", default_priority=1.0)
                    print("✓ Registered with ECAN coordinator for memory attention management")
            except Exception as e:
                print(f"⚠️ OpenCog cognitive memory initialization failed: {e}")
                print("ℹ️ Falling back to basic persistence mode")
                self.load_persistent_memory()  # Load in fallback mode
        else:
            print("⚠️ OpenCog not available - using basic persistence mode")
            self.load_persistent_memory()  # Load in fallback mode
    
    async def execute(self, operation: str, data: dict = None, **kwargs):
        """Operations: store, retrieve, associate, reason"""
        
        # Initialize if needed
        self._initialize_if_needed()
        
        # Operations work in both OpenCog and fallback modes
        if operation == "store":
            return await self.store_knowledge(data)
        elif operation == "retrieve":
            return await self.retrieve_knowledge(data)
        elif operation == "associate":
            return await self.create_associations(data)
        elif operation == "reason":
            return await self.cognitive_reasoning(data)
        elif operation == "status":
            return await self.get_memory_status()
        else:
            return Response(message="Unknown cognitive memory operation. Available: store, retrieve, associate, reason, status", break_loop=False)
    
    async def get_memory_status(self):
        """Get status information about cognitive memory."""
        try:
            if self.initialized:
                # OpenCog mode
                total_atoms = len(self.atomspace)
                concept_nodes = len(self.atomspace.get_atoms_by_type(types.ConceptNode))
                evaluation_links = len(self.atomspace.get_atoms_by_type(types.EvaluationLink))
                inheritance_links = len(self.atomspace.get_atoms_by_type(types.InheritanceLink))
                
                status = {
                    "mode": "OpenCog AtomSpace",
                    "initialized": True,
                    "total_atoms": total_atoms,
                    "concept_nodes": concept_nodes,
                    "evaluation_links": evaluation_links,
                    "inheritance_links": inheritance_links,
                    "memory_file": self.memory_file,
                    "file_exists": files.exists(self.memory_file)
                }
            else:
                # Fallback mode
                total_atoms = len(self._fallback_memory.get("atoms", []))
                concept_nodes = len([a for a in self._fallback_memory.get("atoms", []) if a.get("type") == "ConceptNode"])
                evaluation_links = len([a for a in self._fallback_memory.get("atoms", []) if a.get("type") == "EvaluationLink"])
                inheritance_links = len([a for a in self._fallback_memory.get("atoms", []) if a.get("type") == "InheritanceLink"])
                
                status = {
                    "mode": "Fallback Memory",
                    "initialized": False,
                    "total_atoms": total_atoms,
                    "concept_nodes": concept_nodes,
                    "evaluation_links": evaluation_links,
                    "inheritance_links": inheritance_links,
                    "memory_file": self.memory_file,
                    "file_exists": files.exists(self.memory_file),
                    "opencog_available": OPENCOG_AVAILABLE
                }
            
            return Response(
                message=f"Cognitive memory status: {json.dumps(status)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(message=f"Error getting memory status: {e}", break_loop=False)
    
    async def store_knowledge(self, data: dict):
        """Store Agent-Zero knowledge in AtomSpace or fallback memory."""
        if not data or "concept" not in data:
            return Response(message="Invalid data: 'concept' field required", break_loop=False)
        
        try:
            concept_name = data["concept"]
            
            # Request attention allocation for memory storage
            if ECAN_COORDINATOR_AVAILABLE:
                concepts = [concept_name]
                if "properties" in data:
                    concepts.extend(data["properties"].keys())
                
                priority = data.get("importance", 1.0)  # Use importance if provided
                request_attention_for_tool(
                    tool_name="cognitive_memory",
                    priority=priority,
                    context=f"Storing knowledge: {concept_name}",
                    concepts=concepts[:6],  # Limit concepts
                    importance_multiplier=1.0
                )
            
            if self.initialized:
                # Store in OpenCog AtomSpace
                concept = self.atomspace.add_node(types.ConceptNode, concept_name)
                
                if "properties" in data:
                    for prop, value in data["properties"].items():
                        prop_node = self.atomspace.add_node(types.ConceptNode, prop)
                        value_node = self.atomspace.add_node(types.ConceptNode, str(value))
                        
                        # Create inheritance and evaluation links
                        self.atomspace.add_link(types.InheritanceLink, [concept, prop_node])
                        self.atomspace.add_link(types.EvaluationLink, [prop_node, value_node])
                
                # Store relationships if provided
                if "relationships" in data:
                    for rel_type, targets in data["relationships"].items():
                        if not isinstance(targets, list):
                            targets = [targets]
                        
                        for target in targets:
                            target_node = self.atomspace.add_node(types.ConceptNode, str(target))
                            rel_node = self.atomspace.add_node(types.PredicateNode, rel_type)
                            self.atomspace.add_link(types.EvaluationLink, [rel_node, concept, target_node])
                
            else:
                # Store in fallback memory structure
                self._store_in_fallback_memory(concept_name, data)
            
            self.save_persistent_memory()
            
            return Response(
                message=f"Stored knowledge about '{concept_name}' in cognitive memory",
                break_loop=False
            )
            
        except Exception as e:
            return Response(message=f"Error storing knowledge: {e}", break_loop=False)
    
    def _store_in_fallback_memory(self, concept_name, data):
        """Store knowledge in fallback memory structure when OpenCog is not available."""
        
        # Create concept node
        concept_atom = {
            "type": "ConceptNode",
            "name": concept_name,
            "outgoing": [],
            "created_at": str(__import__('datetime').datetime.now())
        }
        
        # Add concept if it doesn't exist
        existing_concept = self._find_atom_in_fallback(concept_name, "ConceptNode")
        if not existing_concept:
            self._fallback_memory["atoms"].append(concept_atom)
        
        # Add properties
        if "properties" in data:
            for prop, value in data["properties"].items():
                # Add property node
                prop_atom = {
                    "type": "ConceptNode", 
                    "name": prop,
                    "outgoing": [],
                    "created_at": str(__import__('datetime').datetime.now())
                }
                if not self._find_atom_in_fallback(prop, "ConceptNode"):
                    self._fallback_memory["atoms"].append(prop_atom)
                
                # Add value node
                value_atom = {
                    "type": "ConceptNode",
                    "name": str(value),
                    "outgoing": [],
                    "created_at": str(__import__('datetime').datetime.now())
                }
                if not self._find_atom_in_fallback(str(value), "ConceptNode"):
                    self._fallback_memory["atoms"].append(value_atom)
                
                # Add inheritance link
                inheritance_link = {
                    "type": "InheritanceLink",
                    "name": None,
                    "outgoing": [concept_name, prop],
                    "created_at": str(__import__('datetime').datetime.now())
                }
                self._fallback_memory["atoms"].append(inheritance_link)
                
                # Add evaluation link
                eval_link = {
                    "type": "EvaluationLink",
                    "name": None, 
                    "outgoing": [prop, str(value)],
                    "created_at": str(__import__('datetime').datetime.now())
                }
                self._fallback_memory["atoms"].append(eval_link)
        
        # Add relationships
        if "relationships" in data:
            for rel_type, targets in data["relationships"].items():
                if not isinstance(targets, list):
                    targets = [targets]
                
                # Add predicate node
                pred_atom = {
                    "type": "PredicateNode",
                    "name": rel_type,
                    "outgoing": [],
                    "created_at": str(__import__('datetime').datetime.now())
                }
                if not self._find_atom_in_fallback(rel_type, "PredicateNode"):
                    self._fallback_memory["atoms"].append(pred_atom)
                
                for target in targets:
                    # Add target node
                    target_atom = {
                        "type": "ConceptNode",
                        "name": str(target),
                        "outgoing": [],
                        "created_at": str(__import__('datetime').datetime.now())
                    }
                    if not self._find_atom_in_fallback(str(target), "ConceptNode"):
                        self._fallback_memory["atoms"].append(target_atom)
                    
                    # Add evaluation link for relationship
                    rel_link = {
                        "type": "EvaluationLink",
                        "name": None,
                        "outgoing": [rel_type, concept_name, str(target)],
                        "created_at": str(__import__('datetime').datetime.now())
                    }
                    self._fallback_memory["atoms"].append(rel_link)
        
        # Update metadata
        self._fallback_memory["metadata"]["total_atoms"] = len(self._fallback_memory["atoms"])
        self._fallback_memory["metadata"]["last_updated"] = str(__import__('datetime').datetime.now())
    
    def _find_atom_in_fallback(self, name, atom_type):
        """Find an atom in fallback memory by name and type."""
        for atom in self._fallback_memory["atoms"]:
            if atom.get("name") == name and atom.get("type") == atom_type:
                return atom
        return None
    
    async def retrieve_knowledge(self, data: dict):
        """Retrieve knowledge from AtomSpace or fallback memory."""
        if not data or "concept" not in data:
            return Response(message="Invalid data: 'concept' field required for retrieval", break_loop=False)
        
        try:
            concept_name = data["concept"]
            
            if self.initialized:
                # Retrieve from OpenCog AtomSpace
                return await self._retrieve_from_atomspace(concept_name)
            else:
                # Retrieve from fallback memory
                return await self._retrieve_from_fallback(concept_name)
                
        except Exception as e:
            return Response(message=f"Error retrieving knowledge: {e}", break_loop=False)
    
    async def _retrieve_from_atomspace(self, concept_name):
        """Retrieve knowledge from OpenCog AtomSpace."""
        # Find the concept node
        concept_nodes = [atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode) 
                       if atom.name == concept_name]
        
        if not concept_nodes:
            return Response(
                message=f"Concept '{concept_name}' not found in cognitive memory",
                break_loop=False
            )
        
        concept_node = concept_nodes[0]
        
        # Gather related information
        knowledge = {
            "concept": concept_name,
            "properties": {},
            "relationships": {},
            "incoming_links": [],
            "outgoing_links": []
        }
        
        # Get incoming links (relationships where this concept is the target)
        incoming = concept_node.incoming
        for link in incoming:
            if link.type == types.InheritanceLink:
                if len(link.out) == 2 and link.out[1] == concept_node:
                    knowledge["properties"][link.out[0].name] = "inherited_property"
            elif link.type == types.EvaluationLink:
                if len(link.out) >= 2:
                    predicate = link.out[0].name
                    if predicate not in knowledge["relationships"]:
                        knowledge["relationships"][predicate] = []
                    # Extract other participants in the relationship
                    for atom in link.out[1:]:
                        if atom != concept_node:
                            knowledge["relationships"][predicate].append(atom.name)
            
            knowledge["incoming_links"].append(str(link))
        
        # Get outgoing links (if any)
        if hasattr(concept_node, 'outgoing') and concept_node.outgoing:
            knowledge["outgoing_links"] = [str(link) for link in concept_node.outgoing]
        
        return Response(
            message=f"Retrieved knowledge about '{concept_name}' from cognitive memory. Data: {json.dumps(knowledge)}",
            break_loop=False
        )
    
    async def _retrieve_from_fallback(self, concept_name):
        """Retrieve knowledge from fallback memory."""
        # Find the concept
        concept_atom = self._find_atom_in_fallback(concept_name, "ConceptNode")
        if not concept_atom:
            return Response(
                message=f"Concept '{concept_name}' not found in cognitive memory",
                break_loop=False
            )
        
        # Gather related information
        knowledge = {
            "concept": concept_name,
            "properties": {},
            "relationships": {},
            "links": []
        }
        
        # Find all links involving this concept
        for atom in self._fallback_memory["atoms"]:
            if atom.get("outgoing") and concept_name in atom["outgoing"]:
                knowledge["links"].append({
                    "type": atom["type"],
                    "outgoing": atom["outgoing"]
                })
                
                # Parse specific relationship types
                if atom["type"] == "InheritanceLink" and len(atom["outgoing"]) == 2:
                    if atom["outgoing"][1] == concept_name:
                        prop_name = atom["outgoing"][0]
                        knowledge["properties"][prop_name] = "inherited_property"
                
                elif atom["type"] == "EvaluationLink" and len(atom["outgoing"]) >= 2:
                    predicate = atom["outgoing"][0]
                    if predicate not in knowledge["relationships"]:
                        knowledge["relationships"][predicate] = []
                    
                    for target in atom["outgoing"][1:]:
                        if target != concept_name:
                            knowledge["relationships"][predicate].append(target)
        
        total_links = len(knowledge["links"])
        return Response(
            message=f"Retrieved knowledge about '{concept_name}' from cognitive memory ({total_links} connections). Data: {json.dumps(knowledge)}",
            break_loop=False
        )
    
    async def create_associations(self, data: dict):
        """Create associations between concepts in AtomSpace or fallback memory."""
        if not data or "source" not in data or "target" not in data:
            return Response(message="Invalid data: 'source' and 'target' fields required", break_loop=False)
        
        try:
            source_name = data["source"]
            target_name = data["target"]
            association_type = data.get("type", "similar_to")
            
            if self.initialized:
                # Create association in OpenCog AtomSpace
                source_node = self.atomspace.add_node(types.ConceptNode, source_name)
                target_node = self.atomspace.add_node(types.ConceptNode, target_name)
                predicate_node = self.atomspace.add_node(types.PredicateNode, association_type)
                
                # Create the association link
                association_link = self.atomspace.add_link(
                    types.EvaluationLink, 
                    [predicate_node, source_node, target_node]
                )
                
                # Add strength/confidence if provided
                if "strength" in data:
                    try:
                        from opencog.atomspace import TruthValue
                        strength = float(data["strength"])
                        confidence = float(data.get("confidence", 0.9))
                        association_link.tv = TruthValue(strength, confidence)
                    except:
                        pass  # Graceful fallback if TruthValue not available
            else:
                # Create association in fallback memory
                self._create_association_in_fallback(source_name, target_name, association_type, data)
            
            self.save_persistent_memory()
            
            return Response(
                message=f"Created association: {source_name} --{association_type}--> {target_name}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(message=f"Error creating association: {e}", break_loop=False)
    
    def _create_association_in_fallback(self, source_name, target_name, association_type, data):
        """Create association in fallback memory."""
        
        # Add source node if not exists
        if not self._find_atom_in_fallback(source_name, "ConceptNode"):
            source_atom = {
                "type": "ConceptNode",
                "name": source_name,
                "outgoing": [],
                "created_at": str(__import__('datetime').datetime.now())
            }
            self._fallback_memory["atoms"].append(source_atom)
        
        # Add target node if not exists
        if not self._find_atom_in_fallback(target_name, "ConceptNode"):
            target_atom = {
                "type": "ConceptNode",
                "name": target_name,
                "outgoing": [],
                "created_at": str(__import__('datetime').datetime.now())
            }
            self._fallback_memory["atoms"].append(target_atom)
        
        # Add predicate node if not exists
        if not self._find_atom_in_fallback(association_type, "PredicateNode"):
            predicate_atom = {
                "type": "PredicateNode",
                "name": association_type,
                "outgoing": [],
                "created_at": str(__import__('datetime').datetime.now())
            }
            self._fallback_memory["atoms"].append(predicate_atom)
        
        # Create evaluation link
        association_link = {
            "type": "EvaluationLink",
            "name": None,
            "outgoing": [association_type, source_name, target_name],
            "created_at": str(__import__('datetime').datetime.now())
        }
        
        # Add truth value information if provided
        if "strength" in data:
            association_link["truth_value"] = {
                "strength": float(data["strength"]),
                "confidence": float(data.get("confidence", 0.9))
            }
        
        self._fallback_memory["atoms"].append(association_link)
        
        # Update metadata
        self._fallback_memory["metadata"]["total_atoms"] = len(self._fallback_memory["atoms"])
        self._fallback_memory["metadata"]["last_updated"] = str(__import__('datetime').datetime.now())
    
    async def cognitive_reasoning(self, data: dict):
        """Perform basic cognitive reasoning on stored knowledge."""
        try:
            query = data.get("query", "") if data else ""
            
            if self.initialized:
                return await self._reason_with_atomspace(query)
            else:
                return await self._reason_with_fallback(query)
            
        except Exception as e:
            return Response(message=f"Error in cognitive reasoning: {e}", break_loop=False)
    
    async def _reason_with_atomspace(self, query):
        """Reasoning with OpenCog AtomSpace."""
        # Simple reasoning: find related concepts and connections
        reasoning_results = {
            "query": query,
            "total_atoms": len(self.atomspace),
            "concept_nodes": len(self.atomspace.get_atoms_by_type(types.ConceptNode)),
            "evaluation_links": len(self.atomspace.get_atoms_by_type(types.EvaluationLink)),
            "inheritance_links": len(self.atomspace.get_atoms_by_type(types.InheritanceLink)),
            "connected_concepts": []
        }
        
        # If a specific query is provided, find related concepts
        if query:
            query_concepts = [atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode) 
                            if query.lower() in atom.name.lower()]
            
            for concept in query_concepts[:5]:  # Limit to top 5 matches
                concept_info = {
                    "concept": concept.name,
                    "connections": len(concept.incoming)
                }
                reasoning_results["connected_concepts"].append(concept_info)
        
        return Response(
            message=f"Cognitive reasoning completed for query: '{query}'. Results: {json.dumps(reasoning_results)}",
            break_loop=False
        )
    
    async def _reason_with_fallback(self, query):
        """Reasoning with fallback memory structure."""
        atoms = self._fallback_memory.get("atoms", [])
        
        reasoning_results = {
            "query": query,
            "mode": "fallback",
            "total_atoms": len(atoms),
            "concept_nodes": len([a for a in atoms if a.get("type") == "ConceptNode"]),
            "evaluation_links": len([a for a in atoms if a.get("type") == "EvaluationLink"]),
            "inheritance_links": len([a for a in atoms if a.get("type") == "InheritanceLink"]),
            "connected_concepts": []
        }
        
        # If a specific query is provided, find related concepts
        if query:
            query_concepts = [atom for atom in atoms 
                            if atom.get("type") == "ConceptNode" 
                            and atom.get("name") 
                            and query.lower() in atom["name"].lower()]
            
            for concept in query_concepts[:5]:  # Limit to top 5 matches
                concept_name = concept["name"]
                # Count connections by finding links that reference this concept
                connections = len([a for a in atoms 
                                 if a.get("outgoing") and concept_name in a["outgoing"]])
                
                concept_info = {
                    "concept": concept_name,
                    "connections": connections,
                    "created_at": concept.get("created_at", "unknown")
                }
                reasoning_results["connected_concepts"].append(concept_info)
        
        return Response(
            message=f"Cognitive reasoning completed for query: '{query}' (fallback mode). Results: {json.dumps(reasoning_results)}",
            break_loop=False
        )
    
    def load_persistent_memory(self):
        """Load AtomSpace from persistent storage with fallback support."""
        try:
            if files.exists(self.memory_file):
                # Try loading the main memory file
                atomspace_data = self._load_memory_file(self.memory_file)
                if atomspace_data:
                    if self.initialized:
                        self.restore_atomspace(atomspace_data)
                    else:
                        # Store data for fallback mode
                        self._fallback_memory = atomspace_data
                    print(f"✓ Loaded cognitive memory from {self.memory_file}")
                    return
            
            # Try backup file if main file doesn't exist or is corrupted
            backup_file = self.memory_file.replace('.pkl', '_backup.pkl')
            if files.exists(backup_file):
                atomspace_data = self._load_memory_file(backup_file)
                if atomspace_data:
                    if self.initialized:
                        self.restore_atomspace(atomspace_data)
                    else:
                        self._fallback_memory = atomspace_data
                    print(f"✓ Loaded cognitive memory from backup: {backup_file}")
                    return
                    
            print("ℹ️ No existing cognitive memory found - starting fresh")
            
        except Exception as e:
            print(f"⚠️ Could not load cognitive memory: {e}")
            # Initialize empty fallback memory
            self._fallback_memory = {"atoms": [], "metadata": {"total_atoms": 0}}
    
    def _load_memory_file(self, file_path):
        """Load and validate a memory file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Validate the data structure
            if not isinstance(data, dict):
                print(f"⚠️ Invalid memory file format: {file_path}")
                return None
            
            if "atoms" not in data or "metadata" not in data:
                print(f"⚠️ Corrupted memory file - missing required fields: {file_path}")
                return None
            
            return data
            
        except (pickle.PickleError, EOFError, ValueError) as e:
            print(f"⚠️ Corrupted memory file {file_path}: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Error loading memory file {file_path}: {e}")
            return None
    
    def save_persistent_memory(self):
        """Save AtomSpace to persistent storage with backup and validation."""
        try:
            # Ensure memory directory exists
            memory_dir = os.path.dirname(self.memory_file)
            os.makedirs(memory_dir, exist_ok=True)
            
            # Serialize data
            if self.initialized:
                atomspace_data = self.serialize_atomspace()
            else:
                # Use fallback memory if OpenCog not available
                atomspace_data = getattr(self, '_fallback_memory', {"atoms": [], "metadata": {"total_atoms": 0}})
            
            if not atomspace_data or not atomspace_data.get("atoms"):
                print("ℹ️ No cognitive memory data to save")
                return
            
            # Validate data before saving
            if not self._validate_memory_data(atomspace_data):
                print("⚠️ Memory data validation failed - not saving")
                return
            
            # Create backup of existing file
            if files.exists(self.memory_file):
                backup_file = self.memory_file.replace('.pkl', '_backup.pkl')
                try:
                    import shutil
                    shutil.copy2(self.memory_file, backup_file)
                except Exception as backup_error:
                    print(f"⚠️ Could not create backup: {backup_error}")
            
            # Save new data to temporary file first
            temp_file = self.memory_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(atomspace_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Verify the saved file
            try:
                with open(temp_file, 'rb') as f:
                    test_data = pickle.load(f)
                if not self._validate_memory_data(test_data):
                    raise ValueError("Saved data validation failed")
            except Exception as verify_error:
                os.remove(temp_file)
                raise ValueError(f"File verification failed: {verify_error}")
            
            # Atomically replace the old file
            import shutil
            shutil.move(temp_file, self.memory_file)
            
            total_atoms = len(atomspace_data.get("atoms", []))
            print(f"✓ Saved cognitive memory to {self.memory_file} ({total_atoms} atoms)")
            
        except Exception as e:
            print(f"⚠️ Could not save cognitive memory: {e}")
            # Clean up temporary file if it exists
            temp_file = self.memory_file + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def _validate_memory_data(self, data):
        """Validate memory data structure and content."""
        try:
            if not isinstance(data, dict):
                return False
                
            if "atoms" not in data or "metadata" not in data:
                return False
                
            atoms = data["atoms"]
            if not isinstance(atoms, list):
                return False
            
            # Check each atom has required fields
            for atom in atoms:
                if not isinstance(atom, dict):
                    return False
                if "type" not in atom:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def serialize_atomspace(self):
        """Serialize AtomSpace to a dictionary for persistence."""
        if not self.initialized:
            return {}
        
        try:
            serialized_data = {
                "atoms": [],
                "metadata": {
                    "total_atoms": len(self.atomspace),
                    "created_at": str(__import__('datetime').datetime.now())
                }
            }
            
            # Serialize all atoms
            for atom in self.atomspace:
                atom_data = {
                    "type": atom.type,
                    "name": atom.name if hasattr(atom, 'name') else None,
                    "outgoing": [str(out_atom) for out_atom in atom.out] if hasattr(atom, 'out') else []
                }
                
                # Include truth value if available
                if hasattr(atom, 'tv') and atom.tv:
                    try:
                        atom_data["truth_value"] = {
                            "strength": atom.tv.mean,
                            "confidence": atom.tv.confidence
                        }
                    except:
                        pass
                
                serialized_data["atoms"].append(atom_data)
            
            return serialized_data
            
        except Exception as e:
            print(f"Error serializing AtomSpace: {e}")
            return {}
    
    def restore_atomspace(self, atomspace_data):
        """Restore AtomSpace from serialized data."""
        if not self.initialized or not atomspace_data:
            return
        
        try:
            atoms_data = atomspace_data.get("atoms", [])
            
            # First pass: create all nodes
            for atom_data in atoms_data:
                if not atom_data.get("outgoing"):  # This is a node
                    atom_type = atom_data["type"]
                    atom_name = atom_data.get("name", "")
                    
                    if atom_name:
                        node = self.atomspace.add_node(atom_type, atom_name)
                        
                        # Restore truth value if available
                        if "truth_value" in atom_data:
                            try:
                                from opencog.atomspace import TruthValue
                                tv_data = atom_data["truth_value"]
                                node.tv = TruthValue(tv_data["strength"], tv_data["confidence"])
                            except:
                                pass
            
            # Second pass: create all links
            for atom_data in atoms_data:
                if atom_data.get("outgoing"):  # This is a link
                    atom_type = atom_data["type"]
                    outgoing_names = atom_data["outgoing"]
                    
                    # Find the outgoing atoms
                    outgoing_atoms = []
                    for name in outgoing_names:
                        # Simple name-based lookup (could be improved)
                        matching_atoms = [atom for atom in self.atomspace if str(atom) == name]
                        if matching_atoms:
                            outgoing_atoms.append(matching_atoms[0])
                    
                    if outgoing_atoms:
                        link = self.atomspace.add_link(atom_type, outgoing_atoms)
                        
                        # Restore truth value if available
                        if "truth_value" in atom_data:
                            try:
                                from opencog.atomspace import TruthValue
                                tv_data = atom_data["truth_value"]
                                link.tv = TruthValue(tv_data["strength"], tv_data["confidence"])
                            except:
                                pass
            
            print(f"✓ Restored {len(atoms_data)} atoms to cognitive memory")
            
        except Exception as e:
            print(f"Error restoring AtomSpace: {e}")


def register():
    return CognitiveMemoryTool