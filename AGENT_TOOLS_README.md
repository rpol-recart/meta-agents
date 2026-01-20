# Agent Tools Demo and Testing Guide

This directory contains scripts for testing the new agent capabilities.

## Quick Demo (No API Key Required)

Run the demo to see all tools in action without calling the LLM:

```bash
python demo_agent_tools.py
```

This demonstrates:
- **Semantic Analyzer**: Entity extraction, sentiment analysis
- **Pattern Recognizer**: Pattern detection, anti-pattern detection, code metrics
- **Dependency Analyzer**: Import analysis, dependency graph, circular dependency detection

## Full Agent Testing (Requires API Key)

For full agent testing with LLM integration, use:

```bash
# Test semantic analyzer agent
python test_agent_prompt.py --agent semantic_analyzer

# Test pattern recognizer agent with code sample
python test_agent_prompt.py --agent pattern_recognizer --code

# Test dependency analyzer agent with code sample
python test_agent_prompt.py --agent dependency_analyzer --code

# Custom prompt
python test_agent_prompt.py --agent semantic_analyzer --prompt "Your custom prompt here"

# Verbose output
python test_agent_prompt.py --agent pattern_recognizer --code -v

# Save result to file
python test_agent_prompt.py --agent dependency_analyzer --code --output result.json
```

## Available Agents

### semantic_analyzer (9 tools)
- `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- `extract_entities` - Extract named entities (technologies, companies, materials, processes)
- `analyze_sentiment` - Analyze sentiment and strategic implications
- `save_entities_to_neo4j` - Save entities to Neo4j knowledge graph
- `analyze_text_full` - Complete analysis with auto-save to Neo4j

### pattern_recognizer (11 tools)
- `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- `analyze_patterns` - Detect design patterns (Singleton, Factory, Observer, etc.)
- `detect_anti_patterns` - Identify anti-patterns (God Object, Spaghetti Code, etc.)
- `calculate_code_metrics` - Calculate code quality metrics
- `save_patterns_to_neo4j` - Save patterns to Neo4j
- `analyze_code_full` - Complete pattern analysis with auto-save
- `detect_specific_pattern` - Detect a specific pattern by name

### dependency_analyzer (11 tools)
- `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- `find_imports` - Extract all imports (standard lib, third-party, local, relative)
- `analyze_file_dependencies` - Analyze dependencies for a single file
- `build_dependency_graph` - Build a dependency graph from multiple files
- `detect_circular_dependencies` - Detect circular dependencies
- `calculate_dependency_metrics` - Calculate dependency metrics
- `save_dependencies_to_neo4j` - Save dependency graph to Neo4j

## Environment Variables

For Neo4j integration, set these in `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Aura DB (alternative)
AURA_INSTANCEID=your-instance-id
AURA_INSTANCENAME=your-instance-name
```

## Sample Code for Testing

The demo scripts use sample code that includes:

1. **Singleton Pattern** - Classic implementation with `get_instance()`
2. **God Object Anti-pattern** - Class with too many responsibilities
3. **Magic Numbers** - Hard-coded numeric literals
4. **Various Imports** - Standard lib, third-party, local imports
5. **Circular Dependencies** - A → B → C → A pattern

## Integration Testing

To verify tools are working correctly:

```bash
# Run all tool tests
python -m pytest tests/tools/test_analysis_tools.py -v
python -m pytest tests/tools/test_pattern_tools.py -v
python -m pytest tests/tools/test_dependency_tools.py -v

# Run all tests
python -m pytest tests/ -v
```
