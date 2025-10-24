# Development Workflow

## Standard Development Process

For every phase or subtask in this project, follow this workflow:

### 1. ✅ Implement the Code
- Write the necessary code files
- Test that the code works (run imports, basic tests)
- Ensure code quality and documentation

### 2. ✅ Update tasks.md
- Mark completed tasks with `[x]` instead of `[ ]`
- Add ✅ emoji to completed phase headers
- Keep the file up-to-date with current progress

### 3. ✅ Commit with Descriptive Message
- Stage all relevant files (`git add`)
- Write a clear, descriptive commit message
- Use the format:
  ```
  Complete Phase X: [Phase Name]

  - Bullet point summary of changes
  - What was implemented
  - Any important details
  ```
- Commit the changes

## Important Notes

- **Never skip the tasks.md update** - it's our source of truth for progress
- **Commit frequently** - after each phase or significant subtask
- **Write clear commit messages** - they document the project history
- **Test before committing** - ensure code works before marking tasks complete

## Starting a New Session

At the beginning of each session, the AI assistant should:
1. Read this WORKFLOW.md file
2. Check tasks.md to see what's been completed
3. Check git log to see recent commits
4. Understand the current state before proceeding

## Project Structure Reference

- `tasks.md` - Master task list with all implementation phases
- `README.md` - Project documentation
- `config.py` - Configuration constants
- See tasks.md for complete phase breakdown
