# Contributing to Hierarchical Emotional Intelligence

We welcome contributions to the Hierarchical EI project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/hierarchical-ei.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/hierarchical-ei.git
cd hierarchical-ei

# Create virtual environment
conda create -n hierarchical_ei_dev python=3.8
conda activate hierarchical_ei_dev

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run these before committing:

```bash
# Format code
black .

# Check linting
flake8 .

# Type checking
mypy hierarchical_ei/
```

## Testing

All new features should include tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hierarchical_ei tests/

# Run specific test
pytest tests/test_model.py::test_forward_pass
```

## Adding New Features

### 1. Model Components

When adding new model components:

- Place in appropriate module (`models/`, `training/`, etc.)
- Include docstrings with parameter descriptions
- Add unit tests
- Update configuration schema if needed

Example:
```python
class NewComponent(nn.Module):
    """Brief description of component.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        
    Example:
        >>> component = NewComponent(256, 512)
        >>> output = component(input_tensor)
    """
```

### 2. Datasets

For new datasets:

- Inherit from `EmotionDataset` base class
- Implement required methods
- Add download script to `scripts/`
- Document preprocessing steps

### 3. Training Features

New training features should:

- Be configurable via YAML
- Include logging via `logging` module
- Support checkpointing
- Add to command-line interface if applicable

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function(param1: int, param2: str) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input
    """
```

### Updating Documentation

1. Update docstrings in code
2. Update README if adding major features
3. Add examples to `notebooks/`
4. Update API documentation if needed

## Pull Request Process

1. **Title**: Use clear, descriptive titles
2. **Description**: Include:
   - What changes were made
   - Why they were made
   - How to test them
   - Related issues
3. **Tests**: Ensure all tests pass
4. **Documentation**: Update as needed
5. **Review**: Address reviewer feedback

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] No merge conflicts

## Reporting Issues

When reporting issues, include:

1. **Environment**: Python version, PyTorch version, OS
2. **Description**: Clear description of the issue
3. **Reproduction**: Minimal code to reproduce
4. **Expected vs Actual**: What should happen vs what happens
5. **Error messages**: Full traceback if applicable

## Feature Requests

For feature requests:

1. Check existing issues first
2. Provide clear use case
3. Explain why it would benefit the project
4. Be open to discussion

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy

### Unacceptable Behavior

- Harassment or discrimination
- Personal attacks
- Trolling or inflammatory comments
- Publishing private information
- Other unprofessional conduct

## Recognition

Contributors will be recognized in:

- AUTHORS.md file
- Release notes
- Paper acknowledgments (for significant contributions)

## Questions?

Feel free to:

- Open an issue for questions
- Reach out via email
- Join our Discord server (if applicable)

Thank you for contributing to Hierarchical EI!