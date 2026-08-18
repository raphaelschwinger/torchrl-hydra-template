#!/usr/bin/env bash
# Fix intermittent Claude Code VS Code extension login/API failures:
#   getaddrinfo EREFUSED platform.claude.com
#
# Campus/Docker DNS sometimes refuses Anthropic lookups, and IPv6 often
# resolves but cannot connect. Pin Anthropic hosts to current IPv4 via
# Cloudflare DNS-over-HTTPS (HTTPS:443 works even when UDP:53 to public
# resolvers is blocked — do NOT rewrite /etc/resolv.conf with 1.1.1.1 /
# 8.8.8.8; that breaks github.com and everything else on firewalled nets).
set -euo pipefail

MARKER_BEGIN="# BEGIN claude-dns-fix"
MARKER_END="# END claude-dns-fix"
HOSTS=(platform.claude.com claude.ai api.anthropic.com)

resolve_a() {
  local host="$1"
  python3 - "$host" <<'PY'
import json, sys, urllib.request
host = sys.argv[1]
url = f"https://1.1.1.1/dns-query?name={host}&type=A"
req = urllib.request.Request(url, headers={"accept": "application/dns-json"})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.load(resp)
ips = [a["data"] for a in data.get("Answer", []) if a.get("type") == 1]
if not ips:
    raise SystemExit(f"no A record for {host}")
print(ips[0])
PY
}

# --- /etc/hosts: pin Anthropic domains to IPv4 ---
tmp_hosts="$(mktemp)"
if [[ -f /etc/hosts ]]; then
  awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
    $0 == b {skip=1; next}
    $0 == e {skip=0; next}
    !skip {print}
  ' /etc/hosts >"$tmp_hosts"
else
  : >"$tmp_hosts"
fi

{
  echo "$MARKER_BEGIN"
  echo "# Managed by .devcontainer/fix-claude-dns.sh — do not edit by hand"
  for host in "${HOSTS[@]}"; do
    ip="$(resolve_a "$host")"
    echo "$ip $host"
  done
  echo "$MARKER_END"
} >>"$tmp_hosts"

sudo cp "$tmp_hosts" /etc/hosts
rm -f "$tmp_hosts"

# Strip a previous broken prefix of public nameservers if present. Campus
# networks often block UDP/53 to 1.1.1.1 and 8.8.8.8; glibc then times out
# before falling through to the working Docker/campus resolvers.
if [[ -f /etc/resolv.conf ]] && grep -q 'Prefixed by .devcontainer/fix-claude-dns.sh' /etc/resolv.conf; then
  tmp_resolv="$(mktemp)"
  awk '
    /^# Prefixed by \.devcontainer\/fix-claude-dns\.sh/ { next }
    /^nameserver[ \t]+(1\.1\.1\.1|8\.8\.8\.8)([ \t]|$)/ { next }
    { print }
  ' /etc/resolv.conf >"$tmp_resolv"
  sudo cp "$tmp_resolv" /etc/resolv.conf
  rm -f "$tmp_resolv"
fi

echo "claude-dns-fix: pinned ${HOSTS[*]} to IPv4 (left system resolvers alone)"
