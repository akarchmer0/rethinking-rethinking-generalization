# Contributing to Rethinking Generalization

Thank you for your interest in contributing to this research project!

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](https://github.com/yourusername/rethinking-generalization-rebuttal/issues) page
2. If not, create a new issue with:
   - Clear description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU info)

### Contributing Code

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/rethinking-generalization-rebuttal.git
   cd rethinking-generalization-rebuttal
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add docstrings to all functions
   - Include type hints where appropriate
   - Add tests for new functionality

4. **Run Tests**
   ```bash
   pytest tests/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Describe your changes

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable names
- Keep functions focused and modular
- Add comments for complex logic
- Maximum line length: 100 characters

## Testing

- All new features should include tests
- Tests should be in the `tests/` directory
- Use pytest for testing
- Aim for good coverage of edge cases

## Documentation

- Update README.md if adding new features
- Add docstrings to all public functions/classes
- Include usage examples where helpful

## Experiment Guidelines

When adding new experiments:

1. Create a new file in `src/experiments/`
2. Follow the structure of existing experiments
3. Use the centralized configuration from `src/utils/config.py`
4. Save results in a reproducible format
5. Add visualization functions to `src/analysis/visualization.py`
6. Document the experiment in the README

## Research Ethics

- Be honest about results, including negative findings
- Cite all relevant prior work
- Share code and data when possible
- Be respectful in discussions and reviews

## Questions?

Feel free to open an issue for questions or reach out via email.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

