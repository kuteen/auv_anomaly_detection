# Coding standards

1. The best code is the code that is unwritten. Make the smallest change that fully solves the task.
2. No options, wrappers, abstractions or features nobody asked for. Delete before you add.
3. Extract shared code after the second duplication, not before. Shared helpers live in my_modules, import them, never copy them.
4. One idea per function, named for what it does. If it needs a comment, rewrite it. No comments in code, explain in the commit message.
5. Fail fast. Raise explicit errors, never swallow one or fall back silently.
6. No v1, v2, .old or .bak copies. Git history is the archive.
