# Free-tier infrastructure proposal

`main.tf` pins the Cloudflare and Vercel providers. It describes one D1 database per corpus year (2014–2026), with `prevent_destroy`. The Vercel project lookup is optional because this configuration does not modify the existing site. No infrastructure has been applied.

On 3 October 2026, Terraform 1.13.5 `fmt -check`, `init -backend=false`, and `validate` passed. A local `plan -refresh=false` with placeholder account and token values proposed **13 additions, zero changes, zero destroys**. That is a syntax and shape check, not an authenticated plan against the owner's account. Run an authenticated `terraform plan` and review database sizes, service limits, and permissions before considering any apply. The user has reserved `terraform apply` for explicit approval.

Year shards are only a candidate: the downloaded XML measures 787.7 MB and the complete local FTS index 1.86 GB. Separate 2014 and 2026 SQLite shards measured 147.2 MB and 126.0 MB. Cloudflare's free plan caps a single D1 database and account storage, and imposes daily write/read limits. The importer in `scripts/publish_d1.py` has a per-run cap and state, but has not been exercised against a real D1 account. Provisioning a database alone does not make public Q&A available. API credentials, production storage and latency measurements, a reviewed evaluation set, citation checks, and a deployed end-to-end smoke test remain release gates.

The FastAPI rate and daily request limiters are process-local. A public multi-instance deployment needs a shared quota counter or an edge rule that enforces the same limit across instances. Workers AI's provider quota is a final free-tier stop, not a substitute for public abuse control.
