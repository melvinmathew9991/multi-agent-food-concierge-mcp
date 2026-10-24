# deploy

The server image is the repository's [`Dockerfile`](../Dockerfile): the catalog, indexes, embedding model and photo thumbnails are built in, and provider keys and tokens are passed at run time, never baked in.

```bash
docker build -t food-concierge .
docker run --rm -p 7860:7860 --env-file .env food-concierge
python scripts/smoke_deploy.py http://127.0.0.1:7860
```

The `docker` workflow builds the image, starts it and runs the same smoke test on `main`, on PRs that change the image's inputs, and by hand.

**Hugging Face Space (Phase 4, #121, pending the owner).** The Space (Docker SDK, free CPU) will run the exact image built and smoke-tested in CI: a tag pushes it once to GitHub's container registry, and the Space's Dockerfile is `FROM` that digest. Space secrets: `GROQ_API_KEY`, `GEMINI_API_KEY`, `API_TOKEN_AGENT`, `API_TOKEN_ADMIN`, and `ALLOWED_HOSTS` set to the Space's host. No staging Space for now.

Streamlit Community Cloud notes come in Phase 7. Bedrock stays off by default and is stub-tested only.
