name: H4 RSI Divergencia ellenőrzés

on:
  schedule:
    - cron: "*/15 * * * *"   # 15 percenként
  workflow_dispatch:

permissions:
  contents: write

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run divergence check
        env:
          GMAIL_USER: ${{ secrets.GMAIL_USER }}
          GMAIL_PASS: ${{ secrets.GMAIL_PASS }}
          EMAIL_TO:   ${{ secrets.EMAIL_TO }}
        run: python check_divergence.py

      - name: Commit state.json
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add state.json
          git diff --quiet && git diff --staged --quiet || git commit -m "Update state.json"
          git push
