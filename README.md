# custom-agent-harness

# helpers:

python - <<'EOF'
import ssl
win = "".join(ssl.DER_cert_to_PEM_cert(c)
              for c in ssl.create_default_context().get_ca_certs(binary_form=True))
old = open("PATH", encoding="ascii", errors="ignore").read()
open("PATH", "w", encoding="ascii").write(win + old)
print("written")
EOF
