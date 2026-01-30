# Contributing to Stock Market Prediction Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/Muhammad-Farooq-13/stock-market-prediction.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/ -v`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v --cov=src
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Use pytest for testing

## Pull Request Guidelines

1. **Title**: Clear and descriptive
2. **Description**: Explain what and why
3. **Tests**: Include tests for new features
4. **Documentation**: Update README if needed
5. **Code Quality**: Ensure linting passes

## Reporting Issues

When reporting issues, please include:
- Python version
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions or discussions.

Thank you for contributing!
