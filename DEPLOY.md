# DEPLOY — hybrid dev/live environment (local + Oracle A1, git-synced)

**Goal:** live ≡ test, ₹0/month. One **Oracle A1** VM (2 OCPU / 12 GB, PAYG) holds the Breeze-whitelisted
static IP and runs everything that touches Breeze (data / paper / live); your **laptop** does fast
no-Breeze dev (backtests, signals); **git** keeps both identical; **VS Code Remote-SSH** is the editor for
VM-side work. Nautilus's same-strategy-class parity + identical `uv.lock` ⇒ backtest, paper, and live behave the same.

---

## 0. One-time: lock dependencies (reproducible envs)
On either machine, in the repo:
```
uv lock && git add uv.lock && git commit -m "lock deps" && git push
```
Both sides then `uv sync` to the *exact* same versions.

## 1. Create the Oracle A1 VM (one-time)
1. Oracle Cloud → **Billing → Upgrade to Pay-As-You-Go** — free, but it **removes idle-reclaim** (a bot that idles between market events is a reclaim target on a free-only account). Non-negotiable.
2. Compute → Instances → **Create**:
   - Image: **Ubuntu 24.04 LTS**; Shape: **VM.Standard.A1.Flex, 2 OCPU / 12 GB**; Region: **Mumbai (ap-mumbai-1)**; public subnet + assign public IPv4; upload your SSH **public** key (`~/.ssh/id_ed25519.pub`; make one with `ssh-keygen -t ed25519`).
   - *"Out of host capacity"* → retry, switch Availability Domain, or try off-peak. (A1 launch is a lottery.)
3. Networking → convert the instance's public IP to a **Reserved** IP (survives stop/start; ephemeral can change and break your whitelist).
4. Inbound firewall: leave only **SSH (22)** open — the bot makes only **outbound** calls to Breeze, so no app ports are needed.
5. **Never manually stop the instance** (restart can hit "out of capacity" mid-market). Set a **billing alert at ~₹1**.

## 2. Whitelist the IP in Breeze
Breeze app settings → **Primary IP = the A1 reserved IP**. (Editable only **once per week** — get it right.)

## 3. Provision the VM (SSH in once)
```
ssh ubuntu@<A1_RESERVED_IP>
curl -LsSf https://astral.sh/uv/install.sh | sh && exec $SHELL
sudo apt-get update && sudo apt-get install -y git tmux
git clone https://github.com/prateekrastogi/arbitrage.git && cd arbitrage
uv sync                       # exact deps from uv.lock
cp .env.example .env          # fill BREEZE_* (this file is gitignored)
```

## 4. Local laptop (mirror — identical env)
```
git clone https://github.com/prateekrastogi/arbitrage.git && cd arbitrage
uv sync                       # same uv.lock => byte-identical deps to the VM
cp .env.example .env
```

## 5. VS Code Remote-SSH (edit local-feel, execute on the VM)
Add to `~/.ssh/config`:
```
Host arb-vm
    HostName <A1_RESERVED_IP>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
```
VS Code → install **"Remote - SSH"** → Connect to Host → `arb-vm` → Open Folder `~/arbitrage`.
Now editing feels local, but terminal / run / debug all execute **on the VM** → egress is natively the whitelisted IP. **Never** use `ssh -D`/SOCKS for Breeze — its websocket won't honor a proxy and would leak your home IP.

## 6. The daily loop
| Task | Where | Command |
|---|---|---|
| Backtest / signals / cost model (**no Breeze**) | **Local** (fast) | `uv run python -m volarb.run.backtest` |
| Data pull / paper / live (**Breeze**) | **VM** (Remote-SSH) | `uv run python -m volarb.run.live` |
| Sync | git | local `git push` → VM `git pull` |

Same `uv.lock` + same code + Nautilus same-strategy-class ⇒ what you backtest is what runs live.

### Author code in ONE place
Don't edit on both machines. **Author on the VM via Remote-SSH** (Breeze/live run there anyway, and the 12 GB A1 backtests fine); commit/push from the VM. Keep the **laptop as a `git pull`-and-run mirror** — never edit there. One editing copy ⇒ no two-machine conflicts to manage.

## 7. Daily Breeze session token
The token expires at **24 h or midnight** and login is interactive (no headless refresh). Each trading day:
build the login URL (`adapters/breeze/auth.py`), open it in a browser (**any IP** — only API *calls* need the
whitelisted IP, not the browser login), copy the `API_Session` value (DevTools → Network → the `127.0.0.1` row
→ Payload/Form Data), put it in the VM's `.env`, and (re)start the bot.

## 8. Keep the live bot alive (A1, unattended)
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
*Summary: laptop = fast no-Breeze dev; A1 = the whitelisted Breeze/live runner; git = the bridge; Remote-SSH = the editor. PAYG within free limits ⇒ ₹0/month, no idle-reclaim.*
