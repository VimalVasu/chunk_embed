## 2. Environment & Tools

- **Language & Framework**  
  - TypeScript, React (Next.js)  
  - Tailwind CSS, shadcn/ui  
- **Backend Services**  
  - Supabase (Auth, Storage, Postgres)  
  - ChromaDB (vector store)  
- **CLI & Utilities**  
  - `claude` CLI (agentic coding & prompt testing)  
  - Node.js & npm scripts (`npm run dev`, `npm run build`)  
  - FFmpeg (audio processing)  

- **Logging**
  - whenever you run into an error, summarize the erorr along with the solution if found.  place them in [debuglog.md](./debuglog.md) 
- **Task Logging**
  - After a task is complete, mark the task as complete in Tasks.md as [[x]](./Tasks.md), which you would put in the beginning of the task line
- **User input**
  - at points where I need to do some work not in the development environment to get the project progressing, stop and ask me to input any details or do any setting up work so that you can continue to develop.
---

## 5. Allowed Tools & Permissions

By default, Claude will prompt before running:
- **File operations** (`Edit`, `Move`, `Delete`)  
- **Shell commands** (`npm run dev`, `ffmpeg`, `supabase db push`)  

To streamline common tasks, consider pre‑allowing:
```jsonc
// .claude/settings.json
{
  "allowedTools": [
    "Edit", 
    "Bash(npm run dev)", 
    "Bash(ffmpeg *)", 
    "Bash(supabase *)"
  ]
}