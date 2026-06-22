# DEPLOY — dev/live environment (one Oracle A1 VM, via VS Code Remote-SSH)

**One box for everything, ₹0/month.** The Oracle A1 (`vol-arb`, 2 OCPU / 12 GB, PAYG) holds the
Breeze-whitelisted static IP and runs it *all* — editing, backtests, paper, live. You work on it through
**VS Code Remote-SSH**: the editor UI is on your laptop, but the VS Code server, terminal, debugger, and every
process run **on the VM**, so egress is natively the whitelisted IP. Your laptop is just the VS Code client —
**nothing to install or clone locally.** Git pushes to GitHub for versioning/backup. Nautilus's
same-strategy-class parity + the locked `uv.lock` ⇒ backtest = paper = live.

---

## 1. Create the Oracle A1 VM (one-time)
1. Oracle Cloud → **Billing → Upgrade to Pay-As-You-Go** — free, but it **removes idle-reclaim** (a bot that idles between market events is a reclaim target on a free-only account). Non-negotiable.
2. Compute → Instances → **Create**:
   - Image: **Ubuntu 24.04 LTS**; Shape: **VM.Standard.A1.Flex, 2 OCPU / 12 GB**; **Capacity type: On-demand** (not dedicated, not preemptible); Region: **Mumbai (ap-mumbai-1)**; public subnet + assign public IPv4; upload your SSH **public** key (`ssh-keygen -t ed25519`).
   - *"Out of host capacity"* → retry, switch Availability Domain, or try off-peak. (A1 launch is a lottery.)
   - **Sizing:** the free A1 = **2 OCPU / 12 GB** (Oracle halved it from 4/24 in June 2026). On **PAYG you can scale to 4 OCPU / 24 GB**, but only the first 2/12 is free — a full always-on 4/24 bills the overage at ≈ **$27/mo (~₹2,300)**. The free 2/12 is plenty for this bot.
3. Networking → convert the public IP to a **Reserved** IP (survives stop/start; ephemeral can change and break your whitelist).
4. Inbound firewall: leave only **SSH (22)** open — the bot makes only **outbound** calls to Breeze.
5. **Never manually stop the instance** (restart can hit "out of capacity" mid-market). Set a **billing alert at ~₹1**.

## 2. Whitelist the IP in Breeze
Breeze app settings → **Primary IP = the A1 reserved IP**. (Editable only **once per week** — get it right.)

## 3. Provision the VM (one-time, SSH in)
```
ssh ubuntu@<A1_RESERVED_IP>
curl -LsSf https://astral.sh/uv/install.sh | sh && exec $SHELL
sudo apt-get update && sudo apt-get install -y git tmux
# GitHub deploy key, so the PRIVATE repo can clone/pull/push over SSH:
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""    # add ~/.ssh/id_ed25519.pub to the repo →
                                                     # Settings → Deploy keys (tick "Allow write access")
git clone git@github.com:prateekrastogi/arbitrage.git && cd arbitrage
uv sync                                              # installs the exact locked env (uv.lock)
git config user.name "prateekrastogi" && git config user.email "prateek.rastogi@bolster.ai"
cp .env.example .env                                 # fill BREEZE_* (gitignored)
```

## 4. Develop — VS Code Remote-SSH (this *is* the whole environment)
On your laptop, add to `~/.ssh/config`:
```
Host arb-vm
    HostName <A1_RESERVED_IP>
    User ubuntu
    IdentityFile ~/.ssh/oracle_arb_a1
    IdentitiesOnly yes
```
VS Code → install **"Remote - SSH"** → Connect to Host → `arb-vm` → Open Folder `~/arbitrage`.
Edit / run / debug — all on the VM. There is **no local setup** beyond the VS Code client. (Never `ssh -D`/SOCKS for Breeze — its websocket won't honor a proxy.)

## 5. The loop (all on the VM)
- Code in VS Code (running on the VM). Run in the VM terminal:
  - backtest / signals: `uv run python -m volarb.run.backtest`
  - paper / live: `uv run python -m volarb.run.live`
- Versioning / backup: `git add -A && git commit -m "…" && git push` from the VM.
- **One machine ⇒ nothing to sync, no conflicts.**

## 6. Daily Breeze session token
The token expires at **24 h or midnight** and login is interactive (no headless refresh). Each trading day:
build the login URL (`adapters/breeze/auth.py`), open it in a browser (**any IP** — only API *calls* need the
whitelisted IP, not the browser login), copy the `API_Session` value (DevTools → Network → the `127.0.0.1` row
→ Payload/Form Data), put it in the VM's `.env`, and (re)start the bot.

## 7. Keep the live bot alive (unattended)
Run under **systemd** so it survives SSH disconnects and auto-restarts on crash:
```
# /etc/systemd/system/volarb.service
[Unit]
Description=volarb live
After=network-online.target
[Service]
WorkingDirectory=/home/ubuntu/arbitrage
ExecStart=/home/ubuntu/.local/bin/uv run python -m volarb.run.live
Restart=always
User=ubuntu
[Install]
WantedBy=multi-user.target
```
`sudo systemctl enable --now volarb` (or just `tmux` for manual sessions).
**Live-money reliability:** keep a **secondary whitelisted IP**, a **manual square-off** fallback, and (eventually) a ~₹500/mo dedicated VPS once real capital scales.

---
*One Oracle A1 (PAYG, free-tier-sized) = the whitelisted Breeze runner **and** your dev box via Remote-SSH. ₹0/month, no idle-reclaim, and nothing on the laptop but the VS Code client.*
