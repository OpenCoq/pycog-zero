# New AtomSpace Bindings Integration

## Overview

This document describes the new atomspace bindings integrated into `python/tools/cognitive_reasoning.py` as part of Phase 1 (Core Extensions) of the Agent-Zero Genesis development roadmap.

## What's New

### 1. Enhanced Value Types

The updated implementation now supports OpenCog's new Value API, providing richer metadata storage capabilities:

- **FloatValue**: Store numerical vectors and scores
- **StringValue**: Store text descriptions and labels  
- **LinkValue**: Store collections of atoms
- **BoolValue**: Store boolean flags
- **QueueValue**: Store ordered sequences

### 2. Type Constructors API

Cleaner, more Pythonic atom creation using specific type constructors:

```python
from opencog.type_constructors import ConceptNode, PredicateNode, InheritanceLink

# Old way
atom = atomspace.add_node(types.ConceptNode, "machine_learning")

# New way - explicit imports (no wildcards)
atom = ConceptNode("machine_learning")
```

**Note**: We explicitly import only needed type constructors to avoid namespace pollution and maintain code clarity.

### 3. Value Attachment Methods

New methods for attaching rich metadata to atoms:

```python
# Attach numerical scores
score_key = PredicateNode("relevance")
atom.set_value(score_key, FloatValue([0.95, 0.88, 0.92]))

# Retrieve scores
scores = list(atom.get_value(score_key))
```

### 4. EdgeLink Semantic Relationships

More expressive relationship modeling:

```python
relationship = EdgeLink(
    PredicateNode("enables"),
    ListLink(concept_a, concept_b)
)
```

## New Methods in PLNReasoningTool

### attach_metadata_value()

Attaches metadata to an atom using the Value API:

```python
atom = pln_tool.create_probabilistic_atom("AI", strength=0.9, confidence=0.85)
pln_tool.attach_metadata_value(atom, "category", ["technology", "research"])
```

**Parameters:**
- `atom`: The atom to attach metadata to
- `key_name`: Name of the metadata key
- `value_data`: Data to attach (automatically determines Value type)

**Returns:** The atom with attached metadata

### get_metadata_value()

Retrieves metadata from an atom:

```python
categories = pln_tool.get_metadata_value(atom, "category")
# Returns: ["technology", "research"]
```

**Parameters:**
- `atom`: The atom to retrieve metadata from
- `key_name`: Name of the metadata key

**Returns:** Python list of values, or None if not found

### create_reasoning_context_atom()

Creates an atom that encapsulates reasoning context:

```python
context_atom = pln_tool.create_reasoning_context_atom({
    'name': 'research_context',  # Optional, auto-generated if omitted
    'confidence': 0.9,
    'priority': 2.5,
    'concepts': ['AI', 'machine_learning', 'cognition'],
    'inference_rules': ['deduction', 'induction'],
    'timestamp': '2024-12-06T10:00:00Z'
})
```

**Note**: If 'name' is not provided, a deterministic hash-based name is generated using MD5 for predictable behavior across Python sessions.

**Parameters:**
- `context_data`: Dictionary containing context information

**Returns:** An atom with attached context values

### demonstrate_new_bindings()

Demonstrates usage of new binding features (for documentation/testing):

```python
examples = pln_tool.demonstrate_new_bindings()
```

**Returns:** Dictionary with examples of:
- FloatValue attachments
- StringValue attachments
- Links with metadata
- EdgeLink relationships
- Value manipulation

## Updated Methods

### create_probabilistic_atom()

Now uses both FloatValue and TruthValue for probabilistic atoms:

```python
atom = pln_tool.create_probabilistic_atom(
    "machine_learning",
    strength=0.92,
    confidence=0.88
)

# Retrieve truth value as FloatValue
tv_key = PredicateNode("truth-value")
tv = list(atom.get_value(tv_key))  # [0.92, 0.88]
```

### parse_query_to_atoms()

Enhanced to use type constructors and attach metadata:

```python
atoms = cognitive_tool.parse_query_to_atoms(
    "What is machine learning?",
    context={'related_concepts': ['AI', 'neural networks']}
)

# Each atom now has source metadata attached
```

### enhanced_pattern_matching_reasoning()

Now attaches confidence scores to relationships:

```python
results = cognitive_tool.enhanced_pattern_matching_reasoning(atoms, context)

# Each link has confidence attached as FloatValue
for link in results:
    confidence = link.get_value(PredicateNode("confidence"))
```

## Backward Compatibility

All enhancements include fallback modes for environments without OpenCog:

```python
if VALUE_TYPES_AVAILABLE:
    # Use new bindings
    atom = ConceptNode("concept")
    atom.set_value(key, FloatValue([0.9]))
else:
    # Fallback to original implementation
    atom = atomspace.add_node(types.ConceptNode, "concept")
```

The `VALUE_TYPES_AVAILABLE` flag automatically detects binding availability.

## Usage Examples

### Example 1: Attach Numerical Metadata

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

tool = CognitiveReasoningTool(agent)

# Create concept with metadata
concept = ConceptNode("deep_learning")

# Attach relevance scores
relevance_key = PredicateNode("relevance_score")
concept.set_value(relevance_key, FloatValue([0.95, 0.88, 0.92]))

# Retrieve and process
scores = list(concept.get_value(relevance_key))
avg_score = sum(scores) / len(scores)
```

### Example 2: Semantic Relationships

```python
ai = ConceptNode("artificial_intelligence")
ml = ConceptNode("machine_learning")

# Create relationship with metadata
relationship = EdgeLink(
    PredicateNode("specializes"),
    ListLink(ml, ai)
)

# Attach strength
strength_key = PredicateNode("strength")
relationship.set_value(strength_key, FloatValue([0.92]))
```

### Example 3: Reasoning Context

```python
context_atom = pln_tool.create_reasoning_context_atom({
    'confidence': 0.9,
    'priority': 2.5,
    'concepts': ['machine_learning', 'deep_learning'],
    'inference_rules': ['deduction_rule', 'induction_rule']
})

# Retrieve context
confidence = pln_tool.get_metadata_value(context_atom, "confidence")
concepts = pln_tool.get_metadata_value(context_atom, "concepts")
```

### Example 4: Query with Metadata

```python
atoms = tool.parse_query_to_atoms(
    "How does machine learning enable prediction?",
    context={'related_concepts': ['AI', 'statistics']}
)

# Each atom has source metadata
for atom in atoms:
    source = tool.pln_reasoning.get_metadata_value(atom, "source")
    print(f"{atom.name}: {source}")
```

## Testing

Run the test suite to verify new bindings:

```bash
python3 test_new_atomspace_bindings.py
```

The test validates:
- Import of new Value types
- Value API functionality
- Type constructor usage
- Enhanced method presence
- Backward compatibility

## Benefits

1. **Richer Metadata**: Attach vectors, text, and complex data to atoms
2. **Cleaner Code**: Type constructors provide more Pythonic API
3. **Better Performance**: Value API is optimized for bulk data
4. **Declarative Style**: EdgeLink enables semantic relationship modeling
5. **Backward Compatible**: Works with and without new bindings

## Phase 1 Integration

This update fulfills the Phase 1 (Core Extensions) requirement:

> "Update `python/tools/cognitive_reasoning.py` with new atomspace bindings"

Key achievements:
- ✅ Imported new Value types (FloatValue, StringValue, etc.)
- ✅ Added type constructors API
- ✅ Implemented value attachment methods
- ✅ Updated atom creation to use new bindings
- ✅ Maintained backward compatibility
- ✅ Added comprehensive documentation
- ✅ Created test suite

## Next Steps

With Phase 1 complete, the cognitive reasoning tool now has:
- Enhanced metadata storage via Value API
- Cleaner atom creation via type constructors
- Semantic relationship modeling via EdgeLink
- Full backward compatibility with fallbacks

Future phases can build on these foundations for:
- AtomSpace-Rocks performance optimization (Phase 1 continued)
- URE rule-based reasoning (Phase 2)
- ECAN attention allocation (Phase 3)
- Advanced PLN reasoning (Phase 4)

## References

- [OpenCog AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [Value Types](https://wiki.opencog.org/w/Value)
- [Type Constructors](https://wiki.opencog.org/w/Type_constructors)
- [EdgeLink](https://wiki.opencog.org/w/EdgeLink)
- [PyCog-Zero Genesis Roadmap](./AGENT-ZERO-GENESIS.md)
- [Implementation Summary](./IMPLEMENTATION_SUMMARY.md)

## Support

For issues or questions:
1. Check test output: `python3 test_new_atomspace_bindings.py`
2. Review fallback handling in code
3. Ensure OpenCog bindings are installed: `pip install opencog-atomspace opencog-python`
4. Consult OpenCog documentation for Value API details
