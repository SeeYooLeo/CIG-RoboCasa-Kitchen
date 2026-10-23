#!/usr/bin/env bash
# Development certificate for the teleoperation server.
#
# The certificate carries the LAN IP, localhost and 127.0.0.1, so one file
# covers both the adb-reverse path and a direct LAN connection from the
# headset.
set -euo pipefail
umask 077

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="$REPO_DIR/certs"

FORCE=0
ROBOT_IP=""
for argument in "$@"; do
  case "$argument" in
    --force) FORCE=1 ;;
    -h|--help)
      echo "Usage: $0 [IP] [--force]"
      echo "  IP omitted: detect this machine's LAN address."
      exit 0
      ;;
    *) ROBOT_IP="$argument" ;;
  esac
done

detect_ip() {
  # macOS first: Wi-Fi is usually en0, Ethernet adapters land on en1..en7.
  if command -v ipconfig >/dev/null 2>&1; then
    for interface in en0 en1 en2 en3 en4 en5 en6 en7; do
      local address
      address="$(ipconfig getifaddr "$interface" 2>/dev/null || true)"
      if [[ -n "$address" ]]; then
        echo "$address"
        return 0
      fi
    done
  fi
  if command -v hostname >/dev/null 2>&1; then
    local address
    address="$(hostname -I 2>/dev/null | awk '{print $1}')"
    if [[ -n "$address" ]]; then
      echo "$address"
      return 0
    fi
  fi
  if command -v ip >/dev/null 2>&1; then
    ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}'
    return 0
  fi
  return 1
}

if [[ -z "$ROBOT_IP" ]]; then
  ROBOT_IP="$(detect_ip || true)"
  if [[ -z "$ROBOT_IP" ]]; then
    echo "Could not detect a LAN address. Pass the robot IPv4 address explicitly." >&2
    exit 2
  fi
  echo "Detected LAN address: $ROBOT_IP"
fi

if [[ ! "$ROBOT_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Not an IPv4 address: $ROBOT_IP" >&2
  exit 2
fi

if [[ -e "$CERT_DIR/server.key" || -e "$CERT_DIR/quest_ca.key" ]]; then
  if [[ "$FORCE" -ne 1 ]]; then
    echo "Keys already exist in $CERT_DIR. Re-run with --force to replace them." >&2
    exit 1
  fi
  rm -f "$CERT_DIR"/quest_ca.* "$CERT_DIR"/server.*
fi

mkdir -p "$CERT_DIR"
openssl genrsa -out "$CERT_DIR/quest_ca.key" 3072
openssl req -x509 -new -sha256 -days 3650 \
  -key "$CERT_DIR/quest_ca.key" \
  -out "$CERT_DIR/quest_ca.crt" \
  -subj "/CN=Stretch Quest Local CA"
openssl genrsa -out "$CERT_DIR/server.key" 3072
openssl req -new -sha256 \
  -key "$CERT_DIR/server.key" \
  -out "$CERT_DIR/server.csr" \
  -subj "/CN=$ROBOT_IP"
openssl x509 -req -sha256 -days 825 \
  -in "$CERT_DIR/server.csr" \
  -CA "$CERT_DIR/quest_ca.crt" \
  -CAkey "$CERT_DIR/quest_ca.key" \
  -CAcreateserial \
  -out "$CERT_DIR/server.crt" \
  -extfile <(printf 'subjectAltName=IP:%s,DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth\nbasicConstraints=critical,CA:FALSE\n' "$ROBOT_IP")
chmod 600 "$CERT_DIR/quest_ca.key" "$CERT_DIR/server.key"

echo
echo "Server certificate: $CERT_DIR/server.crt"
echo "Private key:        $CERT_DIR/server.key"
echo "Subject alt names:  IP:$ROBOT_IP, DNS:localhost, IP:127.0.0.1"
echo
echo "Open https://$ROBOT_IP:8443/ in Quest Browser. The certificate is"
echo "signed by your local CA. Trust quest_ca.crt on the headset first."
echo "Keep both private keys on this computer; do not copy them to the headset."
