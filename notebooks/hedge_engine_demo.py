# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   jupytext_format_version: '1.5'
# ---

# %% [markdown]
# # Hedge-Engine Interactive Demo
#
# This notebook-script demonstrates how to interact with Hedge-Engine via HTTP
# and via the local Python API.  It is compatible with **Jupytext**, so you can
# open it as a classic `.ipynb` notebook in Colab or JupyterLab while keeping a
# lightweight diff-friendly `.py` representation in Git.
#
# > Tip: run the stack with `docker compose up` first, or point `API_URL` to a
# > running deployment.

# %%
import os
import requests
from pprint import pprint

API_URL = os.getenv("API_URL", "http://localhost:8000")
print("API base →", API_URL)

# Health check
pprint(requests.get(f"{API_URL}/healthz").json())

# %% [markdown]
# ## Hedge sizing request

# %%
payload = {"asset": "BTC", "amount_usd": 100_000, "override_score": 0.7}
resp = requests.post(f"{API_URL}/hedge", json=payload)
pprint(resp.json())

# %% [markdown]
# ## Prometheus metrics

# %%
metrics_text = requests.get(f"{API_URL}/metrics").text
print("First 20 lines of /metrics:\n")
print("\n".join(metrics_text.splitlines()[:20]))

# %% [markdown]
# ## Direct library usage

# %%
from hedge_engine.sizer import compute_hedge

compute_hedge(0.7, depth1pct_usd=5_000_000) 