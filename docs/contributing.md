---
title: Contributing
---

# Contributing to LLMForge

We welcome contributions! This guide will help you get started.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](code-of-conduct.md).

## Ways to Contribute

- 🐛 **Bug Reports** - Report issues on GitHub
- 💡 **Feature Requests** - Suggest new features
- 📝 **Documentation** - Improve docs, examples, tutorials
- 💻 **Code** - Submit Pull Requests
- 🎯 **Testing** - Help test new features
- 🌐 **Translations** - Help translate documentation

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ZandrixAI/llmforge.git
cd llmforge

# Install in development mode
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## Coding Standards

- Follow **PEP 8** style guide
- Use **type hints** where possible
- Write **docstrings** for all public functions
- Add **tests** for new features
- Keep changes **focused and atomic**

## Pull Request Process

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/amazing-feature`)
3. Make your **changes** with tests
4. Ensure all tests **pass** (`pytest`)
5. Update **documentation** if needed
6. **Commit** with clear messages
7. **Push** to your fork
8. Open a **Pull Request**

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear
- [ ] PR description explains the change

## Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(inference): add streaming support for generate()

Added stream parameter to generate() method for real-time token streaming.
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_generate.py

# Run with coverage
pytest --cov=llmforge
```

## Documentation

- Use clear, concise language
- Include code examples
- Keep docs consistent with existing style

## Getting Help

- **GitHub Issues** - For bugs and feature requests
- **GitHub Discussions** - For questions
- **Discord** - Join our community

---

*Thank you for contributing to LLMForge!*