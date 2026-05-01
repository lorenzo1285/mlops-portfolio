# AGENTS.md

- Set agent context: `.specify/scripts/powershell/update-agent-context.ps1 -AgentType claude`
- Update `UBIQUITOUS_LANGUAGE.md` with new terms before tasks (see `plan.md` Pre-Tasks Actions Required)
- Verify Constitution gates via `plan.md`'s Constitution Check table
- Run DVC stages: `dvc repro [stage]`
- Start MLflow UI: `mlflow ui`
- Submit Katib HPO: `kubectl apply -f k8s/katib/vae_experiment.yaml`
- Use `uv` for Python dependencies: `uv pip install -r requirements.txt`
- Refer to `CLAUDE.md` for architecture details