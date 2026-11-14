# Gemini setup for beginners

Gemini is one supported language model provider. The assignment does not require
Gemini specifically; it requires an intelligent agent that selects tools from
natural language. Gemini reads the question and proposes structured tool calls.
Python executes the actual DVC/MLflow/Airflow/Kubernetes tools and produces the
evidence-based result. A paid model is not a requirement.

1. Visit [Google AI Studio — API keys](https://aistudio.google.com/apikey) and sign in.
2. Choose **Create API key**. Create/select a Google project if prompted. Follow
   Google's account requirements; do not enable paid billing just for this guide.
3. Copy the key. Treat it as a password.
4. In VS Code Explorer, open `.env` in this project's root folder. It is a different
   file from the public template `.env.example`. If needed, right-click the folder
   and choose **New File**, then name it `.env`.
5. Find the existing `GEMINI_API_KEY=` line and paste the key after the equals sign.
   Do not put it in Python source, a screenshot, GitHub, or a chat message.
6. Keep these settings and press **Ctrl+S**:

   ```env
   GEMINI_API_KEY=YOUR_ACTUAL_KEY_GOES_HERE
   GEMINI_MODEL=gemini-2.5-flash
   LLM_PROVIDER=gemini
   ```

7. Open a PowerShell terminal in the project root and run:

   ```powershell
   .\.venv\Scripts\python.exe -m agent.main 'Which datasets are being tracked?' --mode llm --provider gemini
   ```

8. The response should show `"mode": "llm"` and actual `dvc_tracked` or
   `dvc_current` evidence with `"status": "ok"`. Try a different wording to show
   language understanding, for example **"Has our training data changed?"**.
9. With MLflow, Airflow and Kubernetes running, try the required combined request:

   ```powershell
   .\.venv\Scripts\python.exe -m agent.main 'Find the best recent model, identify the dataset version used to train it, check deployment status, and explain whether retraining is necessary.' --mode llm --provider gemini
   ```

10. Export the actual conversations with `python -m scripts.capture_evidence --mode llm`.
    A missing or unavailable service stays an error, even if Gemini itself works.

The CLI uses `.env` automatically. Existing operating-system variables take
precedence. With a configured key, the default CLI mode automatically uses the LLM;
`--mode rules` explicitly selects the small deterministic demo router. To use local
Ollama instead, specify `--mode llm --provider ollama`.

If the planner is unavailable: check that the key was saved without spaces, the key
belongs to the intended project, the configured model is available to that project,
and the project has available quota. Do not print the key while debugging. Revoke
and replace a key if it was exposed. The repository ignores `.env` automatically;
`git check-ignore .env` should print `.env`.

If Google says **403 / free tier not available for this project**, the key cannot
complete this demonstration under that project's current eligibility. This is a
provider account restriction. Use an eligible project you already control or
`--mode llm --provider ollama` to demonstrate the local alternative.

Official reference: [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling).
