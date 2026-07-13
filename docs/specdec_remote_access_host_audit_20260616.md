# SpecDec Remote Access Host Audit

Summary: failed=1, failed_auth_or_mfa=2, failed_dns=2, ok=6, timeout=3
Remote squeue: not_checked=8, ok=6

| host | configured hostname | user | dns | controlmaster | connect | remote | squeue | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cw-dfw-cs-001-vscode-01 | `cw-dfw-cs-001-vscode-01` | `sna` | `ok` | `no_controlmaster` | `ok` | `cw-dfw-cs-001-vscode-01`/`sna` | `ok` | cw-dfw-cs-001-vscode-01; sna; __SPECDEC_SQUEUE_OK__ |
| cw-dfw-cs-001-vscode-02 | `cw-dfw-cs-001-vscode-02` | `sna` | `ok` | `no_controlmaster` | `timeout` | ``/`` | `` | timed out after 9s |
| cw-dfw-cs-001-vscode-02.nvidia.com | `cw-dfw-cs-001-vscode-02.nvidia.com` | `sna` | `ok` | `no_controlmaster` | `timeout` | ``/`` | `` | timed out after 9s |
| login-lyris | `login-lyris` | `sna-mfa` | `ok` | `ok` | `ok` | `login-lyris02.lyris.clusters.nvidia.com`/`sna` | `ok` | login-lyris02.lyris.clusters.nvidia.com; sna; __SPECDEC_SQUEUE_OK__ |
| login-lyris.nvidia.com | `login-lyris.nvidia.com` | `sna` | `ok` | `failed` | `failed_auth_or_mfa` | ``/`` | `` | ################################################################################ |
| login-lyris01 | `login-lyris01` | `sna` | `ok` | `failed` | `failed_auth_or_mfa` | ``/`` | `` | ################################################################################ |
| login-lyris02 | `login-lyris02` | `sna` | `ok` | `failed` | `failed` | ``/`` | `` | Host key verification failed. |
| lyris | `lyris` | `sna` | `failed_dns` | `failed` | `failed_dns` | ``/`` | `` | ssh: Could not resolve hostname lyris: nodename nor servname provided, or not known |
| lyris.nvidia.com | `lyris.nvidia.com` | `sna` | `failed_dns` | `failed` | `failed_dns` | ``/`` | `` | ssh: Could not resolve hostname lyris.nvidia.com: nodename nor servname provided, or not known |
| oci-hsg-cs-001-vscode-01 | `oci-hsg-cs-001-vscode-01` | `sna` | `ok` | `ok` | `ok` | `oci-hsg-cs-001-vscode-01`/`sna` | `ok` | oci-hsg-cs-001-vscode-01; sna; __SPECDEC_SQUEUE_OK__ |
| oci-hsg-cs-001-vscode-02 | `oci-hsg-cs-001-vscode-02` | `sna` | `ok` | `no_controlmaster` | `timeout` | ``/`` | `` | timed out after 9s |
| oci-hsg-cs-001-vscode-02.nvidia.com | `oci-hsg-cs-001-vscode-02.nvidia.com` | `sna` | `ok` | `failed` | `ok` | `oci-hsg-cs-001-vscode-02`/`sna` | `ok` | oci-hsg-cs-001-vscode-02; sna; __SPECDEC_SQUEUE_OK__ |
| oci-hsg-cs-001-vscode-03 | `oci-hsg-cs-001-vscode-03` | `sna` | `ok` | `no_controlmaster` | `ok` | `oci-hsg-cs-001-vscode-03`/`sna` | `ok` | oci-hsg-cs-001-vscode-03; sna; __SPECDEC_SQUEUE_OK__ |
| oci-nrt-cs-001-login-03 | `oci-nrt-cs-001-login-03` | `sna` | `ok` | `no_controlmaster` | `ok` | `oci-nrt-cs-001-login-03`/`sna` | `ok` | oci-nrt-cs-001-login-03; sna; __SPECDEC_SQUEUE_OK__ |
