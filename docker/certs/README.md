# docker/certs/

Gitignored, dev-only. `seshat-api` runs in a Linux container; `USE_OS_TRUSTSTORE` sources
trust from the OS the *process* runs on (the container's Debian), not the Windows host —
so the corporate proxy's root CA has to be copied in separately.

Regenerate `zscaler-root-ca.crt` after it rotates or on a new machine:

```powershell
$cert = Get-ChildItem -Path Cert:\LocalMachine\Root | Where-Object { $_.Subject -like '*Zscaler Root CA*' }
[System.IO.File]::WriteAllText('docker/certs/zscaler-root-ca.crt', "-----BEGIN CERTIFICATE-----`n" + [Convert]::ToBase64String($cert.RawData, 'InsertLineBreaks') + "`n-----END CERTIFICATE-----`n")
```

`docker-compose.override.yml` mounts it into `/usr/local/share/ca-certificates/` and runs
`update-ca-certificates` before starting the API.
