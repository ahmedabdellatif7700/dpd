# The testbed's clocks

Status: **set up on the Pi and the Jetson, 2026-11-21.** A record of what was changed on each
machine, by whom, why, how it was checked, and how to undo it.

## Why

Every stage stamps the frames it handles (`t_recv_ns`, `t_send_ns`, against the source's
`capture_time_ns`; on-chip-training PIPELINE.md, Section 7). Between two machines those stamps
compare only as well as the two clocks agree. Before this change each machine followed a different
public pool with `systemd-timesyncd`, polling every 34 minutes:

| | Before | Root distance |
|---|---|---|
| Pi | `systemd-timesyncd`, `2.arch.pool.ntp.org`, stratum 2 | 9 ms |
| Jetson | `systemd-timesyncd`, `1.pool.ntp.org`, stratum 3 | 19 ms |

Each was "synchronized", but to a different server, to within milliseconds: the two could disagree
by tens of milliseconds, against per-stage latencies of a few milliseconds. `systemd-timesyncd` is
also a client only: it cannot serve another machine. The Jetson's network interface has no hardware
time stamps (`ethtool -T`: software only), so PTP would run in software, with no gain over NTP here.

**The choice: chrony on both, the Pi serving the testbed's subnet and the Jetson following the Pi
only.** What the latency logs need is the two clocks' agreement with each other over the local
cable, not their distance to world time.

## What was changed

### Pi (Arch Linux ARM) — by the user

```bash
sudo pacman -S --noconfirm chrony
grep -qE '^(pool|server)' /etc/chrony.conf || echo 'pool 2.arch.pool.ntp.org iburst' | sudo tee -a /etc/chrony.conf
printf '\n# serve the Jetson (dpd hookup)\nallow 192.168.91.0/25\nlocal stratum 10\n' | sudo tee -a /etc/chrony.conf
sudo systemctl disable --now systemd-timesyncd
sudo systemctl enable --now chronyd
```

- chrony installed; Arch's default `/etc/chrony.conf` already has `pool 2.arch.pool.ntp.org iburst`,
  so the second line added nothing;
- appended to `/etc/chrony.conf`: `allow 192.168.91.0/25` (serve the subnet: the Jetson) and
  `local stratum 10` (keep serving from its own clock if the internet is gone, so the Jetson stays
  locked to the Pi);
- `systemd-timesyncd` stopped and disabled at boot (two services must not steer one clock);
  `chronyd` started and enabled at boot.

### Jetson (Ubuntu 24.04) — by Claude, with the user's go-ahead

Run with `sudo -n` (passwordless sudo on the Jetson).

```bash
sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y chrony     # replaces systemd-timesyncd
sudo -n cp -n /etc/chrony/chrony.conf /etc/chrony/chrony.conf.orig   # the original, kept
sudo -n sed -i -E 's/^(pool .*)$/# \1    # off: the Jetson follows the Pi (arof-pi.conf)/;
                   s|^(sourcedir /run/chrony-dhcp)$|# \1    # off: the Jetson follows the Pi (arof-pi.conf)|' \
    /etc/chrony/chrony.conf
sudo -n tee /etc/chrony/conf.d/arof-pi.conf      # below
sudo -n systemctl restart chrony
```

- chrony 4.5 installed; Ubuntu stopped and replaced `systemd-timesyncd` (the packages conflict);
- `/etc/chrony/chrony.conf`: the four `pool` lines (`ntp.ubuntu.com`, `0..2.ubuntu.pool.ntp.org`)
  and `sourcedir /run/chrony-dhcp` (servers from DHCP) commented out, each marked why: the Pi is the
  Jetson's only source; the original is `/etc/chrony/chrony.conf.orig`;
- `/etc/chrony/conf.d/arof-pi.conf`, one active line:

  ```
  server 192.168.91.80 iburst prefer minpoll 2 maxpoll 4
  ```

  follow the Pi, asking every 4 to 16 s (a `makestep 0.1 3` line was added, then removed: Ubuntu's
  `chrony.conf` sets `makestep 1 3` after it reads the drop-in).

## How it was checked

On the Pi (`chronyc tracking`, `chronyc sources`): synchronized to `time.cloudflare.com`, with
`ntp1.tecnico.ulisboa.pt` and `ntp02.oal.ul.pt` also in reach; 4.7 µs from NTP time; listening on
UDP 123.

On the Jetson, after 12 samples over 44 s (`chronyc sourcestats`, `chronyc tracking`):

| | |
|---|---|
| Source | `192.168.91.80` (the Pi), the only one, selected (`*`) |
| Offset to the Pi | mean ~0, standard deviation **2.2 µs** |
| The Jetson's clock | 0.5 µs from the Pi's time; frequency corrected by 29 ppm (its crystal runs slow) |

SSH between the two still works both ways (`ssh rpi4b`, `ssh orinnano8s`): key logins do not depend
on the clock, and chrony did not step it (the clocks were already within milliseconds; it slewed).

Before every run whose latencies are analysed, and after it, record `chronyc tracking` on the
Jetson (and `chronyc sourcestats`), so the logs carry the clocks' agreement with them.

## How to undo

```bash
# Jetson
sudo cp /etc/chrony/chrony.conf.orig /etc/chrony/chrony.conf
sudo rm /etc/chrony/conf.d/arof-pi.conf
sudo systemctl restart chrony          # chrony on the public pools again
#   or back to systemd-timesyncd: sudo apt remove chrony && sudo apt install systemd-timesyncd

# Pi
#   remove the two appended lines from /etc/chrony.conf (allow ..., local stratum 10), or:
sudo systemctl disable --now chronyd && sudo systemctl enable --now systemd-timesyncd
```

## Important, for later: the KV260 too

Every machine whose time stamps are compared needs its clock locked to the same source: the Pi, the
Jetson **and the KV260** (on-chip-training PIPELINE.md, Section 11). Not needed yet for the board:
the Pi sends it version 1 frames, which carry no time stamp. It is needed as soon as the board link
carries version 2 frames (`capture_time_ns`), for the latency from the source to the board: the
KV260 then runs a chrony client of the Pi, as the Jetson does
(`server 192.168.91.80 iburst prefer minpoll 2 maxpoll 4`), and its offset is checked with
`chronyc tracking` before every timed run.
