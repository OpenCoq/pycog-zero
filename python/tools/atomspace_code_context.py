"""
PyCog-Zero AtomSpace Code Execution Context Tool
Stores and analyzes code execution context in OpenCog AtomSpace for cognitive code understanding
"""

from python.helpers.tool import Tool, Response
import json
import ast
import re

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - code context analysis will be limited")
    OPENCOG_AVAILABLE = False


class AtomSpaceCodeContext(Tool):
    """Store and analyze code execution context using AtomSpace."""
    
    def __init__(self, agent, name: str, method: str | None, args: dict, message: str, loop_data, **kwargs):
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self._code_atomspace_initialized = False
        self.atomspace = None
        self.initialized = False
        self.execution_counter = 0
    
    def _initialize_code_atomspace(self):
        """Initialize AtomSpace for code context analysis."""
        if self._code_atomspace_initialized:
            return
        
        self._code_atomspace_initialized = True
        
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.initialized = True
                print("✓ AtomSpace code context analysis initialized")
            except Exception as e:
                print(f"⚠️ AtomSpace code context initialization failed: {e}")
    
    async def execute(self, operation: str = "store_context", code: str = "", result: str = "", language: str = "python", **kwargs):
        """Code context operations: store_context, analyze_patterns, suggest_improvements, trace_dependencies"""
        
        self._initialize_code_atomspace()
        
        if operation == "store_context":
            return await self.store_code_execution_context(code, result, language, **kwargs)
        elif operation == "analyze_patterns":
            return await self.analyze_code_patterns()
        elif operation == "suggest_improvements":
            return await self.suggest_code_improvements(code, language)
        elif operation == "trace_dependencies":
            return await self.trace_code_dependencies(code, language)
        else:
            return Response(
                message="Unknown code context operation. Available: store_context, analyze_patterns, suggest_improvements, trace_dependencies",
                break_loop=False
            )
    
    async def store_code_execution_context(self, code: str, result: str, language: str = "python", **kwargs):
        """Store code execution context in AtomSpace for cognitive analysis."""
        if not self.initialized:
            return Response(
                message="AtomSpace not available - code context storage skipped",
                break_loop=False
            )
        
        try:
            self.execution_counter += 1
            execution_id = f"exec_{self.execution_counter}"
            
            # Create execution concept
            exec_concept = self.atomspace.add_node(types.ConceptNode, execution_id)
            
            # Store language information
            lang_concept = self.atomspace.add_node(types.ConceptNode, f"language_{language}")
            self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, "uses_language"),
                    exec_concept,
                    lang_concept
                ]
            )
            
            # Analyze and store code structure
            code_analysis = await self.analyze_code_structure(code, language)
            for concept_name, concept_type in code_analysis.items():
                concept_node = self.atomspace.add_node(types.ConceptNode, concept_name)
                type_node = self.atomspace.add_node(types.ConceptNode, concept_type)
                
                # Link execution to code concepts
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "contains_concept"),
                        exec_concept,
                        concept_node
                    ]
                )
                
                # Link concept to its type
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "has_type"),
                        concept_node,
                        type_node
                    ]
                )
            
            # Store execution result analysis
            if result:
                result_analysis = await self.analyze_execution_result(result)
                for result_concept in result_analysis:
                    result_node = self.atomspace.add_node(types.ConceptNode, result_concept)
                    self.atomspace.add_link(
                        types.EvaluationLink,
                        [
                            self.atomspace.add_node(types.PredicateNode, "produces_result"),
                            exec_concept,
                            result_node
                        ]
                    )
            
            # Store execution metadata
            metadata = {
                "code_length": len(code),
                "result_length": len(result) if result else 0,
                "execution_order": self.execution_counter
            }
            
            for key, value in metadata.items():
                meta_concept = self.atomspace.add_node(types.ConceptNode, f"{key}_{value}")
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "execution_metadata"),
                        exec_concept,
                        meta_concept
                    ]
                )
            
            return Response(
                message=f"Code execution context stored in AtomSpace. "
                       f"Execution ID: {execution_id}, "
                       f"Code concepts: {len(code_analysis)}, "
                       f"Total atoms: {len(self.atomspace)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error storing code context: {e}",
                break_loop=False
            )
    
    async def analyze_code_structure(self, code: str, language: str):
        """Analyze code structure and extract concepts."""
        concepts = {}
        
        try:
            if language.lower() == "python":
                concepts.update(await self.analyze_python_code(code))
            elif language.lower() in ["javascript", "nodejs"]:
                concepts.update(await self.analyze_javascript_code(code))
            else:
                # Generic analysis for other languages
                concepts.update(await self.analyze_generic_code(code))
                
            return concepts
            
        except Exception as e:
            print(f"Warning: Code structure analysis failed: {e}")
            return {}
    
    async def analyze_python_code(self, code: str):
        """Analyze Python code structure using AST."""
        concepts = {}
        
        try:
            # Parse AST
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    concepts[f"function_{node.name}"] = "function_definition"
                elif isinstance(node, ast.ClassDef):
                    concepts[f"class_{node.name}"] = "class_definition"
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        concepts[f"import_{alias.name}"] = "import_statement"
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        concepts[f"import_{node.module}"] = "import_from_statement"
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            concepts[f"variable_{target.id}"] = "variable_assignment"
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        concepts[f"call_{node.func.id}"] = "function_call"
                    elif isinstance(node.func, ast.Attribute):
                        concepts[f"call_{node.func.attr}"] = "method_call"
        
        except Exception as e:
            print(f"Warning: Python AST analysis failed: {e}")
            # Fallback to regex analysis
            concepts.update(await self.analyze_generic_code(code))
        
        return concepts
    
    async def analyze_javascript_code(self, code: str):
        """Analyze JavaScript code structure using regex patterns."""
        concepts = {}
        
        try:
            # Function definitions
            func_pattern = r'function\s+(\w+)\s*\('
            for match in re.finditer(func_pattern, code):
                concepts[f"function_{match.group(1)}"] = "function_definition"
            
            # Arrow functions with names
            arrow_func_pattern = r'(?:const|let|var)\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
            for match in re.finditer(arrow_func_pattern, code):
                concepts[f"arrow_function_{match.group(1)}"] = "arrow_function"
            
            # Variable declarations
            var_pattern = r'(?:const|let|var)\s+(\w+)\s*='
            for match in re.finditer(var_pattern, code):
                concepts[f"variable_{match.group(1)}"] = "variable_declaration"
            
            # Class definitions
            class_pattern = r'class\s+(\w+)\s*{'
            for match in re.finditer(class_pattern, code):
                concepts[f"class_{match.group(1)}"] = "class_definition"
            
            # Imports
            import_pattern = r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]'
            for match in re.finditer(import_pattern, code):
                concepts[f"import_{match.group(1)}"] = "import_statement"
                
        except Exception as e:
            print(f"Warning: JavaScript analysis failed: {e}")
        
        return concepts
    
    async def analyze_generic_code(self, code: str):
        """Generic code analysis using simple patterns."""
        concepts = {}
        
        try:
            # Simple keyword detection
            keywords = ["function", "class", "def", "import", "from", "var", "let", "const"]
            for keyword in keywords:
                count = len(re.findall(r'\b' + keyword + r'\b', code, re.IGNORECASE))
                if count > 0:
                    concepts[f"keyword_{keyword}"] = f"keyword_usage_{count}"
            
            # Extract identifiers (simple heuristic)
            identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]+\b'
            identifiers = re.findall(identifier_pattern, code)
            
            # Count unique identifiers
            unique_identifiers = set(identifiers)
            for identifier in list(unique_identifiers)[:20]:  # Limit to avoid too many
                concepts[f"identifier_{identifier}"] = "code_identifier"
                
        except Exception as e:
            print(f"Warning: Generic code analysis failed: {e}")
        
        return concepts
    
    async def analyze_execution_result(self, result: str):
        """Analyze execution result and extract meaningful concepts."""
        result_concepts = []
        
        try:
            # Check for common result patterns
            if "error" in result.lower() or "exception" in result.lower():
                result_concepts.append("execution_error")
                
                # Extract error types
                error_pattern = r'(\w+Error|\w+Exception)'
                errors = re.findall(error_pattern, result)
                for error in errors:
                    result_concepts.append(f"error_type_{error}")
            
            elif result.strip().isdigit():
                result_concepts.append("numeric_result")
            
            elif result.strip().startswith(('[', '{')):
                result_concepts.append("structured_data_result")
            
            elif len(result.strip()) > 100:
                result_concepts.append("verbose_output")
            
            else:
                result_concepts.append("simple_output")
            
            # Extract meaningful words from result
            words = re.findall(r'\b[a-zA-Z]{3,}\b', result)
            unique_words = list(set(words))[:10]  # Limit to 10 unique words
            
            for word in unique_words:
                result_concepts.append(f"result_contains_{word.lower()}")
                
        except Exception as e:
            print(f"Warning: Result analysis failed: {e}")
        
        return result_concepts
    
    async def analyze_code_patterns(self):
        """Analyze patterns in stored code executions."""
        if not self.initialized:
            return Response(
                message="AtomSpace not available - pattern analysis skipped",
                break_loop=False
            )
        
        try:
            patterns = []
            
            # Analyze execution frequency
            exec_concepts = [
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if atom.name.startswith("exec_")
            ]
            patterns.append(f"Total executions: {len(exec_concepts)}")
            
            # Analyze language usage
            lang_usage = {}
            for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                if (len(link.out) >= 3 and 
                    hasattr(link.out[0], 'name') and 
                    link.out[0].name == "uses_language"):
                    lang = link.out[2].name if hasattr(link.out[2], 'name') else "unknown"
                    lang_usage[lang] = lang_usage.get(lang, 0) + 1
            
            if lang_usage:
                patterns.append(f"Language usage: {json.dumps(lang_usage)}")
            
            # Analyze common function/concept usage
            concept_frequency = {}
            for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                if (len(link.out) >= 3 and 
                    hasattr(link.out[0], 'name') and 
                    link.out[0].name == "contains_concept"):
                    concept = link.out[2].name if hasattr(link.out[2], 'name') else "unknown"
                    concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
            
            # Top 5 most used concepts
            top_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_concepts:
                patterns.append(f"Top concepts: {json.dumps(dict(top_concepts))}")
            
            # Analyze error patterns
            error_count = len([
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if "error" in atom.name.lower()
            ])
            patterns.append(f"Error instances: {error_count}")
            
            return Response(
                message=f"Code pattern analysis completed:\n" + "\n".join(patterns),
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in pattern analysis: {e}",
                break_loop=False
            )
    
    async def suggest_code_improvements(self, code: str, language: str):
        """Suggest code improvements based on AtomSpace knowledge."""
        if not self.initialized:
            return Response(
                message="AtomSpace not available - improvement suggestions limited",
                break_loop=False
            )
        
        try:
            suggestions = []
            
            # Analyze current code
            current_analysis = await self.analyze_code_structure(code, language)
            
            # Compare with historical patterns
            concept_frequency = {}
            for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                if (len(link.out) >= 3 and 
                    hasattr(link.out[0], 'name') and 
                    link.out[0].name == "contains_concept"):
                    concept = link.out[2].name if hasattr(link.out[2], 'name') else "unknown"
                    concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
            
            # Suggest based on patterns
            if language.lower() == "python":
                if not any("import_" in concept for concept in current_analysis):
                    suggestions.append("Consider adding import statements for better modularity")
                
                if not any("function_" in concept for concept in current_analysis) and len(code) > 50:
                    suggestions.append("Consider breaking code into functions for better organization")
            
            # Check for common error patterns
            error_concepts = [concept for concept in concept_frequency if "error" in concept]
            if error_concepts:
                suggestions.append(f"Be aware of common errors: {', '.join(error_concepts[:3])}")
            
            # Suggest based on successful patterns
            successful_patterns = [
                concept for concept, count in concept_frequency.items() 
                if count > 2 and "error" not in concept
            ]
            if successful_patterns:
                suggestions.append(f"Consider using successful patterns: {', '.join(successful_patterns[:3])}")
            
            if not suggestions:
                suggestions.append("No specific improvement suggestions based on current knowledge")
            
            return Response(
                message=f"Code improvement suggestions for {language}:\n" + "\n".join(f"• {s}" for s in suggestions),
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error generating improvement suggestions: {e}",
                break_loop=False
            )
    
    async def trace_code_dependencies(self, code: str, language: str):
        """Trace code dependencies using AtomSpace relationships."""
        if not self.initialized:
            return Response(
                message="AtomSpace not available - dependency tracing limited",
                break_loop=False
            )
        
        try:
            dependencies = []
            
            # Analyze current code for imports and dependencies
            code_analysis = await self.analyze_code_structure(code, language)
            
            imports = [concept for concept in code_analysis if concept.startswith("import_")]
            functions = [concept for concept in code_analysis if concept.startswith("function_")]
            variables = [concept for concept in code_analysis if concept.startswith("variable_")]
            
            dependencies.append(f"📦 Imports detected: {len(imports)}")
            for imp in imports[:5]:  # Show top 5
                dependencies.append(f"  • {imp}")
            
            dependencies.append(f"🔧 Functions detected: {len(functions)}")
            for func in functions[:5]:  # Show top 5
                dependencies.append(f"  • {func}")
            
            dependencies.append(f"📊 Variables detected: {len(variables)}")
            for var in variables[:5]:  # Show top 5
                dependencies.append(f"  • {var}")
            
            # Find related concepts in AtomSpace
            related_executions = []
            for concept_name in list(code_analysis.keys())[:5]:  # Limit processing
                concept_node = None
                for atom in self.atomspace.get_atoms_by_type(types.ConceptNode):
                    if hasattr(atom, 'name') and atom.name == concept_name:
                        concept_node = atom
                        break
                
                if concept_node:
                    for link in concept_node.incoming:
                        if (link.type == types.EvaluationLink and 
                            len(link.out) >= 3 and 
                            hasattr(link.out[0], 'name') and 
                            link.out[0].name == "contains_concept"):
                            exec_concept = link.out[1]
                            if hasattr(exec_concept, 'name') and exec_concept.name.startswith("exec_"):
                                related_executions.append(exec_concept.name)
            
            if related_executions:
                dependencies.append(f"🔗 Related executions: {len(set(related_executions))}")
            
            return Response(
                message="Code dependency trace:\n" + "\n".join(dependencies),
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error tracing dependencies: {e}",
                break_loop=False
            )


def register():
    """Register the AtomSpace code context tool with Agent-Zero."""
    return AtomSpaceCodeContext