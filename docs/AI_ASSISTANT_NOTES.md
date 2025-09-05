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

## Coding Conventions and Style Preferences

### Column Naming Standards
**CRITICAL: Follow strict snake_case conventions for all internal data processing**

**Internal Data (ALL code, processing, analysis):**
- Use **snake_case** for all column names in DataFrames, variables, and data structures
- Examples: `date`, `close`, `high`, `low`, `volume`, `adjusted_close`
- Database columns (BigQuery) use snake_case format
- All modeling and analysis code should use snake_case

**UI Display DataFrames and Elements:**
- Convert to **Title Case** for DataFrames that appear directly in the UI (st.dataframe, st.table)
- Convert to **Title Case** for user-facing chart elements (axes, legends, labels)
- Examples: `Date`, `Close`, `High`, `Low`, `Volume`, `Adjusted Close` 
- Use `prepare_data_for_display()` function in `chart_layout.py` for conversion
- Apply Title Case conversion only at the final display step, not during processing

**Implementation Guidelines:**
- Internal processing: `data["close"]`, `data["date"]`, `data["volume"]`
- UI display DataFrames: Convert using utility function before displaying with st.dataframe() or st.table()
- Chart display: Convert using utility function before displaying in charts
- Never mix naming conventions within the same operation
- Data pipeline maintains snake_case throughout processing chain

**Price Column Priority for Modeling:**
- Prioritize unadjusted `close` over `adjusted_close` for swing trading analysis
- Use `close` column for all modeling and forecasting
- Only use `adjusted_close` if `close` is unavailable

### Code Organization Principles
- Maintain separation between data logic (snake_case) and presentation (Title Case)
- Use utility functions for consistent naming conversions
- Apply Title Case conversion only at the final display step for UI DataFrames and charts
- Document any deviations from snake_case convention with clear reasoning

## Environment Setup Requirements

### Development Environment Activation
**CRITICAL: Always use environment activation scripts to prevent recurring issues**

Before any development work or running commands:
1. **Use activation script**: Run `activate_env.bat` or `.\activate_env.ps1`
2. **Verify environment**: Script will check conda environment, Python packages, and BigQuery connection
3. **Confirm ready state**: Look for "Ready for development!" message

Available activation methods:
- **Batch script**: `activate_env.bat` (double-click or run in terminal)
- **PowerShell script**: `.\activate_env.ps1` (for VS Code integration)
- **Environment check**: `python check_env.py` (verification only)

### Common Commands After Activation
- `python -m streamlit run main.py` - Launch Streamlit application
- `python scripts/initial_bulk_load.py --help` - Bulk loading options
- `python check_env.py` - Verify environment setup

**Note**: This prevents the recurring issue of commands failing due to missing packages or wrong Python environment.

## Workflow Requirements

### Chat Session Startup
When user says "read the notes file" or references this document:
1. Read this file completely
2. Review current project status from MODERNIZATION_ROADMAP.md
3. **Remind about environment activation** if user plans to run commands
4. Provide brief status summary focusing on next steps
5. Ask what specific task they want to work on

### Chat Session Closing
When user indicates they are closing the chat:
1. **Clean Up Temporary Files**: Delete or move test files, demo scripts, and temporary development files to maintain workspace organization
   - **Debug Scripts**: Review and clean up the `debug_scripts/` folder - delete temporary debugging files, move useful diagnostic scripts to archive, keep only essential development tools
   - **Test Files**: Remove any temporary test files created during development
   - **Demo Scripts**: Clean up any demonstration or proof-of-concept scripts
2. **Check Script Length**: If any scripts we worked on exceed 300 lines, ask user if they want to refactor them for better maintainability
3. **Create Commit Message**: Generate a descriptive commit message for all changes made during the session (PROVIDE THE ACTUAL COMMIT MESSAGE TEXT IN A CODE BLOCK IN CHAT FOR EASY COPYING - DO NOT CREATE FILES FOR COMMIT MESSAGES)
4. **Update Roadmap**: Modify docs/MODERNIZATION_ROADMAP.md to reflect progress and status changes (ALREADY COMPLETED IF DONE DURING SESSION)
5. **Update Requirements**: Update requirements.txt with any new Python packages installed (CHECK IF ANY NEW PACKAGES WERE ADDED)
6. **Environment Variables**: Update .env.example if new environment variables were added (CHECK IF ANY NEW ENV VARS WERE ADDED)
7. **Summary**: Provide brief summary of accomplishments and next recommended steps

IMPORTANT: Session details and continuation context should ONLY go in MODERNIZATION_ROADMAP.md, NOT in this AI_ASSISTANT_NOTES.md file.

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
