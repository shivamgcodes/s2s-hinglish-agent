# S2S serverless: Hugging Face Docker-SDK Space (launcher a) / any container host (launcher c). DESIGN.md 5.2.
# Build context = space/ alone. Before building or pushing:
#   bash space/stage_common.sh            (copies ../common/{s2s_token.py,session.py,data/records_v4.json,data/scripts_v4.json}
#                                          + ../../packages/personaplex_lora/role_prompt.py -> space/common)
#   bash ops/build_client.sh              (client/dist -> space/static)
# Secrets (RUNPOD_API_KEY, S2S_SESSION_SECRET, S2S_PASSCODE) are read from env at runtime only (HF Space Secrets);
# nothing secret is baked into this image.
FROM python:3.11-slim

# [HF-DOCKER] "Permissions": run as uid 1000 with a writable home
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
RUN test -f common/s2s_token.py && test -f common/session.py && test -f common/role_prompt.py && test -f common/data/records_v4.json \
    && test -f common/data/scripts_v4.json \
    || (echo "space/common is missing: run space/stage_common.sh before building" >&2; exit 1)

EXPOSE 7860
CMD ["python", "-u", "-m", "app.main"]
