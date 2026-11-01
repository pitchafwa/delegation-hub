#!/usr/bin/env bash
# Laptop-free trial, day one (2026-11-01): copy this PC's model data into Cloudflare R2 (bucket delegation-hub-data, seed/), where the cloud test
# (cloud-test.yml, cloud-test-dff.yml) starts from. Uses wrangler's own login, so no storage keys on the PC. About 1.3 GB, sent in 250 MB parts
# (wrangler's limit per file is about 300 MB). Run from ingest/:  bash research/seed_cloud_test.sh
set -euo pipefail
B=delegation-hub-data
T=$(mktemp -d)
cd research
tar -czf - data | split -b 250m -d - "$T/data.tar.gz.part"
tar -czf "$T/dfs.tar.gz" -C data dfs
for f in "$T"/data.tar.gz.part* "$T/dfs.tar.gz"; do
  echo "uploading $(basename "$f") ($(du -h "$f" | cut -f1))"
  npx --yes wrangler r2 object put "$B/seed/$(basename "$f")" --file "$f" --remote >/dev/null
done
rm -rf "$T"
echo "seed uploaded"
