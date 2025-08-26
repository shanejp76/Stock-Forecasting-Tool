# AI Assistant Operating Manual

## Communication Style Preferences

### Code Formatting
- **NO EMOJIS** in any code files, comments, code blocks, or documentation files
- Use clean, professional code and documentation without decorative symbols
- Exception: Emojis allowed in chat responses for clarity, but never in actual files
- Documentation should use clear text markers like "DONE:", "TODO:", "COMPLETED:", etc.

### Chat Response Format
- **Exclude code blocks** from explanations unless specifically requested
- Focus on clear explanations and next steps
- Use tools to implement changes rather than showing code
- When code is necessary, use tools like replace_string_in_file or create_file

### Abbreviation Policy
- **Always explain abbreviations** on first use in each conversation
- Examples:
  - API (Application Programming Interface)
  - BigQuery (Google BigQuery data warehouse)
  - CI/CD (Continuous Integration/Continuous Deployment)
  - dbt (data build tool)
  - OHLC (Open, High, Low, Close price data)

### Technical Terminology
- Always define technical terms when first mentioned
- Provide context for domain-specific concepts
- Explain the "why" behind technical decisions

## Workflow Requirements

### Chat Session Startup
When user says "read the notes file" or references this document:
1. Read this file completely
2. Review current project status from MODERNIZATION_ROADMAP.md
3. Provide brief status summary focusing on next steps
4. Ask what specific task they want to work on

### Chat Session Closing
When user indicates they are closing the chat:
1. **Create Commit Message**: Generate a descriptive commit message for all changes made during the session
2. **Update Roadmap**: Modify docs/MODERNIZATION_ROADMAP.md to reflect progress and status changes
3. **Update Requirements**: Update requirements.txt with any new Python packages installed
4. **Environment Variables**: Update .env.example if new environment variables were added
5. **Summary**: Provide brief summary of accomplishments and next recommended steps

### Change Management
- Always use tools to make file changes rather than displaying code
- Test changes when possible using available testing tools
- Update relevant documentation files when making structural changes

## Tool Usage Priority
1. Use semantic_search for finding relevant code
2. Use read_file for understanding context  
3. Use replace_string_in_file for targeted edits
4. Use create_file for new components
5. Use run_in_terminal for testing and execution

---
**Note**: This file should be updated as preferences evolve.
