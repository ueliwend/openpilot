#!/usr/bin/env python3
import fcntl
import json
import logging
import os
import re
import select
import signal
import string
import struct
import subprocess
import tempfile
import termios
import time

from contextlib import contextmanager
from ipaddress import IPv4Address, AddressValueError

from enum import Enum

from openpilot.common.esim.base import WEBBING_ICCID_PREFIX
from openpilot.common.swaglog import cloudlog

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s.%(msecs)03d %(levelname)-7s modem: %(message)s",
  datefmt="%H:%M:%S",
)

AT_PORT = "/dev/modem_at0"
PPP_PORT = "/dev/modem_at1"
STATE_PATH = "/dev/shm/modem"
AT_LOCK = "/dev/shm/modem.lock"  # shared with LPA
AT_INIT = [
  "ATE0",       # disable command echo
  "ATV1",       # verbose result codes (CONNECT/BUSY/NO CARRIER, not numeric)
  "AT+CMEE=1",  # numeric +CME ERROR codes on failures (per 3GPP 27.007)
  "ATX4",       # extended result codes: busy + dial tone detection, line speed in CONNECT
  "AT&C1",      # DCD pin follows carrier state (V.250 default)
  "AT+CREG=2",  # registration URCs include location info
  "AT+CGREG=2", # GPRS registration URCs include location info
]
CREG = {0: "not_registered", 1: "home", 2: "searching", 3: "denied", 4: "unknown", 5: "roaming"}
# 3GPP TS 27.007 +COPS <AcT> -> network type
NETWORK_TYPE = {0: "gsm", 1: "gsm", 3: "gsm", 8: "gsm",
                2: "utran", 4: "utran", 5: "utran", 6: "utran",
                7: "lte", 9: "lte", 10: "lte",
                11: "nr", 12: "nr", 13: "nr"}
# +CPIN: <code> (3GPP TS 27.007) -> SIM state in the modem_status event; any other code is "other"
CPIN = {"READY": "ready", "SIM PIN": "pin_required", "SIM PUK": "puk_required"}
# eUICC ISD-R applet, same value as openpilot.common.esim.lpa.ISDR_AID (not imported: the daemon stays free of the LPA stack)
ISDR_AID = "A0000005591010FFFFFFFF8900000100"
# modem_status logs the firmware version only if the whole reply line has this form, else "" (an unsolicited +CREG / +CGREG
# report read in place of the AT+GMR reply would otherwise carry LAC / TAC and cell id)
VERSION_FORMAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}")
# modem_status is repeated once per drive this long after going onroad: loggerd (onroad only) is recording by then
ONROAD_REPEAT_DELAY = 10.0

DIAL_CID = 1

PPPD_CMD = [
  "sudo", "pppd", PPP_PORT, "460800", "noauth", "nodetach", "noipdefault", "usepeerdns",
  "nodefaultroute", "connect",
  "/usr/sbin/chat -v ABORT 'NO CARRIER' ABORT 'NO DIALTONE' ABORT 'BUSY' " +
  f"ABORT 'NO ANSWER' ABORT 'ERROR' TIMEOUT 5 '' AT OK ATD*99***{DIAL_CID}# CONNECT ''",
  "lcp-echo-interval", "30", "lcp-echo-failure", "4", "mtu", "1500", "mru", "1500",
  "novj", "novjccomp", "ipcp-accept-local", "ipcp-accept-remote", "nomagic",
  "user", '""', "password", '""',
]
INITIAL_STATE: dict[str, object] = {
  "seconds_since_boot": 0,
  "state": "INITIALIZING",
  "connected": False, "ip_address": "",
  "iccid": "", "mcc_mnc": "", "imei": "", "modem_version": "",
  "signal_strength": 0, "signal_quality": 0,
  "network_type": "unknown", "operator": "", "band": "", "channel": 0,
  "registration": "unknown", "temperatures": [], "extra": "",
  "tx_bytes": 0, "rx_bytes": 0,
}


@contextmanager
def _serial_port(port: str, baudrate: int):
  fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
  try:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
    attrs[3] = 0
    attrs[4] = attrs[5] = getattr(termios, f"B{baudrate}")
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    yield fd
  finally:
    os.close(fd)


def _read_line(fd: int, timeout: float) -> bytes:
  data = bytearray()
  deadline = time.monotonic() + timeout
  while True:
    readable, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
    if not readable:
      return bytes(data)
    byte = os.read(fd, 1)
    if not byte:
      return bytes(data)
    data.extend(byte)
    if byte == b"\n":
      return bytes(data)


class State(Enum):
  INITIALIZING = "INITIALIZING"
  SEARCHING = "SEARCHING"
  CONNECTING = "CONNECTING"
  CONNECTED = "CONNECTED"
  DISCONNECTING = "DISCONNECTING"


STATE_WAIT = 1.0  # seconds to wait after each state handler returns


class PPPSession:
  """Owns pppd lifecycle, fail tracking, and PPP routing."""
  MAX_FAILS = 3

  def __init__(self):
    self._proc: subprocess.Popen | None = None
    self._fails = 0
    self._peer = ""

  def start(self):
    self._proc = subprocess.Popen(PPPD_CMD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    self._peer = ""
    logging.info(f"PPP dialing CID {DIAL_CID}")

  def kill(self):
    subprocess.run(["sudo", "killall", "-9", "pppd"], capture_output=True)
    self._peer = ""

  @staticmethod
  def reset_data_port():
    """Drop DTR on PPP_PORT so the modem terminates any stuck PPP session."""
    try:
      with _serial_port(PPP_PORT, 460800) as fd:
        dtr = struct.pack("I", termios.TIOCM_DTR)
        fcntl.ioctl(fd, termios.TIOCMBIC, dtr)
        time.sleep(0.2)
        fcntl.ioctl(fd, termios.TIOCMBIS, dtr)
    except Exception as e:
      logging.warning(f"data port reset failed: {e}")

  def has_exited(self) -> bool:
    return self._proc is not None and self._proc.poll() is not None

  def reset_fail_counter(self):
    self._fails = 0

  def record_fail(self) -> bool:
    """Bump fail counter; return True if at the give-up limit."""
    self._fails += 1
    return self._fails >= self.MAX_FAILS

  @property
  def fails(self) -> int:
    return self._fails

  def maybe_install_routes(self, ip: str, peer: str) -> bool:
    """Install routes if peer changed; kill the session on failure so the state machine reconnects."""
    if not peer or peer == self._peer:
      return False
    try:
      IPv4Address(ip)
      IPv4Address(peer)
    except AddressValueError:
      logging.warning(f"refusing route install with non-IPv4 ip={ip!r} peer={peer!r}")
      self.kill()
      return False
    self.cleanup_routes()
    cmds = [
      ["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "metric", "1000"],
      ["sudo", "ip", "route", "add", "default", "via", peer, "dev", "ppp0", "table", "1000"],
      ["sudo", "ip", "rule", "add", "from", ip, "table", "1000"],
    ]
    for cmd in cmds:
      r = subprocess.run(cmd, capture_output=True, text=True)
      if r.returncode != 0:
        logging.warning(f"route install failed ({' '.join(cmd[1:])}): {r.stderr.strip()}")
        self.cleanup_routes()
        self.kill()
        return False
    logging.info(f"route set up for {ip} via {peer}")
    self._peer = peer
    return True

  def maybe_install_dns(self, dns_servers: list[str]) -> bool:
    """Register DNS servers with systemd-resolved; kill the session on failure to force a retry."""
    if not dns_servers:
      return False
    for cmd in (["sudo", "resolvectl", "dns", "ppp0", *dns_servers],
                ["sudo", "resolvectl", "default-route", "ppp0", "yes"]):
      r = subprocess.run(cmd, capture_output=True, text=True)
      if r.returncode != 0:
        logging.warning(f"resolvectl failed ({' '.join(cmd[1:])}): {r.stderr.strip()}")
        self.kill()
        return False
    logging.info(f"resolvectl: ppp0 DNS = {dns_servers}")
    return True

  @staticmethod
  def cleanup_routes():
    subprocess.run(["sudo", "ip", "route", "del", "default", "dev", "ppp0"], capture_output=True)
    subprocess.run(["sudo", "ip", "route", "flush", "table", "1000"], capture_output=True)
    # rules don't have a flush; delete until none remain
    while subprocess.run(["sudo", "ip", "rule", "del", "table", "1000"], capture_output=True).returncode == 0:
      pass
    subprocess.run(["sudo", "resolvectl", "revert", "ppp0"], capture_output=True)


class Modem:
  def __init__(self):
    self._ppp = PPPSession()
    self._sim_change = False
    self._apn = ""  # blank = network-provided via PCO
    self._roaming_allowed = True
    self.running = True
    self.S = INITIAL_STATE.copy()
    # modem_status event (see _log_identity / _log_registration); _identity_seen keeps the last identity read in memory only
    self._identity_seen: tuple | None = None
    self._sim_status: dict = {}
    self._probe_pending = False  # the last SIM probe found AT_LOCK busy: retried at the next trigger
    self._reg_status = {"registration": "unknown", "data_registration": "unknown", "technology": "unknown"}
    self._onroad_since: float | None = None
    self._onroad_logged = False

  @staticmethod
  def _read_param(key):
    try:
      with open(f"/data/params/d/{key}") as f:
        return f.read().strip()
    except FileNotFoundError:
      return ""

  @staticmethod
  def _parse_reg(v: str) -> str:
    try:
      return CREG.get(int(v.split(",")[1].strip('"')), "unknown")
    except (ValueError, IndexError):
      return "unknown"

  @staticmethod
  def _has_modem_manager() -> bool:
    return os.path.isfile("/lib/systemd/system/ModemManager.service")

  def _is_roaming_allowed(self) -> bool:
    if self.S["iccid"].startswith(WEBBING_ICCID_PREFIX):
      return True
    return self._read_param("GsmRoaming") == "1"

  def _publish_state(self, **kwargs):
    self.S.update(kwargs)
    self.S["seconds_since_boot"] = time.monotonic()
    with tempfile.NamedTemporaryFile(mode="w", dir="/dev/shm", delete=False) as f:
      json.dump(self.S, f, indent=2)
    os.chmod(f.name, 0o644)
    os.replace(f.name, STATE_PATH)

  @contextmanager
  def _at_lock(self):
    """Hold AT_LOCK (shared with the LPA and qcomgpsd) without waiting; yields False if another process holds it."""
    fd = os.open(AT_LOCK, os.O_CREAT | os.O_RDWR, 0o666)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
      locked = True
    except OSError:
      locked = False
    try:
      yield locked
    finally:
      if locked:
        fcntl.flock(fd, fcntl.LOCK_UN)
      os.close(fd)

  def _at(self, cmd):
    """Send AT command, return response lines. [] on error or if LPA holds port."""
    with self._at_lock() as locked:
      return (self._at_send(cmd) or []) if locked else []

  def _at_send(self, cmd) -> list[str] | None:
    """Send AT command without taking AT_LOCK (the caller holds it); response lines, None on error."""
    try:
      with _serial_port(AT_PORT, 9600) as serial_fd:
        termios.tcflush(serial_fd, termios.TCIFLUSH)
        command = (cmd + "\r").encode()
        while command:
          command = command[os.write(serial_fd, command):]
        lines = []
        while True:
          raw = _read_line(serial_fd, 5)
          if not raw:
            raise TimeoutError("AT timeout")
          line = raw.decode(errors="ignore").strip()
          if not line:
            continue
          if line == "OK":
            break
          if line == "ERROR" or line.startswith("+CME ERROR"):
            raise RuntimeError(line)
          lines.append(line)
        return lines
    except (RuntimeError, TimeoutError, OSError) as e:
      logging.info(f"AT {cmd} failed: {e}")
      return None

  def _atv(self, cmd, pfx):
    for line in self._at(cmd):
      if pfx in line and ":" in line:
        return line.split(":", 1)[1].strip()
    return None

  def _init_at_channel(self) -> bool:
    """Run AT_INIT and confirm ATE0 took effect. Returns False if echo is still on."""
    for c in AT_INIT:
      self._at(c)
    r = self._at("AT+CGMI")
    return bool(r) and not r[0].startswith("AT")

  def _configure_modem(self, modem_version: str):
    if not modem_version.startswith("EG25"):
      return
    cmds = [
      # clear initial EPS bearer APN (some carriers reject the default)
      'AT+CGDCONT=0,"IP",""',

      # SIM hot swap
      'AT+QSIMDET=1,0',
      'AT+QSIMSTAT=1',

      # configure modem as data-centric
      'AT+QNVW=5280,0,"0102000000000000"',
      'AT+QNVFW="/nv/item_files/ims/IMS_enable",00',
      'AT+QNVFW="/nv/item_files/modem/mmode/ue_usage_setting",01',
    ]
    for c in cmds:
      self._at(c)

  def _do_initializing(self):
    if not os.path.exists(AT_PORT):
      return State.INITIALIZING
    logging.info("port found, initializing")
    self._ppp.kill()
    self._ppp.cleanup_routes()

    if not self._init_at_channel():
      logging.warning("AT echo still on, retrying")
      return State.INITIALIZING

    identity = self._read_identity()
    self._log_identity(identity)
    if not identity["iccid"] or not identity["imei"]:
      logging.warning(f"identity read incomplete: {identity}, retrying")
      return State.INITIALIZING

    self._configure_modem(identity["modem_version"])

    self.S.update(identity)
    self._apn = self._read_param("GsmApn")
    self._roaming_allowed = self._is_roaming_allowed()
    # blank APN lets the carrier supply one via PCO
    self._at(f'AT+CGDCONT={DIAL_CID},"IP","{self._apn}"')
    logging.info(f"APN '{self._apn or '(network-provided)'}' written to CID {DIAL_CID}, roaming={'on' if self._roaming_allowed else 'off'}")

    self._sim_change = False  # clear since we just re-read identity with the new SIM
    self._publish_state(**identity)
    return State.SEARCHING

  def _read_identity(self):
    def first_line(cmd):  # the first reply line that is not an unsolicited report (+CREG: ..., +QSIMSTAT: ...)
      return next((line.strip() for line in self._at(cmd) if not line.startswith("+")), "")

    imei = first_line("AT+CGSN")
    if not (imei.isdigit() and 14 <= len(imei) <= 17):  # 3GPP TS 23.003
      imei = ""

    iccid = (self._atv("AT+QCCID", "+QCCID:") or "").rstrip("F")
    if not all(c in string.hexdigits for c in iccid):
      iccid = ""

    imsi = first_line("AT+CIMI")
    mcc_mnc = imsi[:6] if imsi.isdigit() and len(imsi) >= 6 else ""

    modem_version = first_line("AT+GMR")

    logging.info(f"imei={imei} iccid={iccid} mcc_mnc={mcc_mnc} ver={modem_version}")
    return {"imei": imei, "iccid": iccid, "mcc_mnc": mcc_mnc, "modem_version": modem_version}

  # modem_status: a cloudlog event (error level, so it reaches qlogs as errorLogMessage, and rlogs) at each new identity read,
  # each registration change and once per drive after going onroad. It holds only booleans, values from the fixed tables
  # above, the SIM's home network code (first 5 IMSI digits: MCC + 2 MNC digits) and the firmware version if it matches
  # VERSION_FORMAT. Never the IMEI, the card id (ICCID), the rest of the IMSI, a phone number or a location: the LAC / TAC
  # and cell id fields of +CREG / +CGREG replies and reports are never stored.

  @staticmethod
  def _clean_version(raw: str, imei: str, iccid: str) -> str:
    """The AT+GMR line as it is if it is a plain version string, else "": no character stripping, no run of 8 digits, not
    containing the IMEI or card id."""
    if not VERSION_FORMAT.fullmatch(raw) or re.search(r"[0-9]{8}", raw):
      return ""
    if any(len(x) >= 6 and x in raw for x in (imei, iccid)):
      return ""
    return raw

  def _log_identity(self, identity: dict):
    """Once at the first identity read the modem answers, then whenever the result changes (SIM inserted, removed, swapped),
    and when a SIM probe that found AT_LOCK busy succeeds at a later read."""
    try:
      if not (identity["imei"] or identity["modem_version"]):
        return  # the modem did not answer (or the LPA holds the port); the read is retried
      iccid = identity["iccid"]
      version = self._clean_version(identity["modem_version"], identity["imei"], iccid)
      seen = (identity["imei"], iccid, identity["mcc_mnc"], version)
      new = seen != self._identity_seen
      if not new and not self._probe_pending:
        return
      self._identity_seen = seen
      if new:
        self._sim_status = {
          "modem_version": version,
          "sim_present": bool(iccid),
          "sim_state": "unknown",
          "euicc": None,
          "comma_profile": iccid.startswith(WEBBING_ICCID_PREFIX) if iccid else None,
          "sim_home_mcc_mnc": "",  # sim_state, euicc and sim_home_mcc_mnc: filled in by _update_probe
        }
        self._reg_status = dict.fromkeys(self._reg_status, "unknown")
      if not self._update_probe() and not new:
        return  # same identity, lock still busy: nothing new to report
      self._emit_status("identity")
    except Exception:
      logging.exception("modem_status (identity) failed")

  def _update_probe(self) -> bool:
    """Fill sim_state, euicc and sim_home_mcc_mnc from _probe_sim. If AT_LOCK is busy they read "unknown" / None / the
    identity read's value, and the probe is retried at the next trigger instead of keeping a wrong value for the boot."""
    _, iccid, mcc_mnc, _ = self._identity_seen
    r = self._probe_sim(bool(iccid), mcc_mnc)
    self._probe_pending = r is None
    self._sim_status.update(r or {"sim_state": "unknown", "euicc": None, "sim_home_mcc_mnc": mcc_mnc[:5]})
    return r is not None

  def _probe_sim(self, card: bool, mcc_mnc: str) -> dict | None:
    """SIM state (AT+CPIN?), eUICC (AT+CCHO / AT+CCHC) and, if the identity read got none, the home network (AT+CIMI), all
    under one hold of AT_LOCK, so that no other user of the port (LPA, qcomgpsd) comes between the commands.
    None if the lock is busy or the probe failed."""
    try:
      with self._at_lock() as locked:
        if not locked:
          return None
        v = next((line.split(":", 1)[1].strip() for line in self._at_send("AT+CPIN?") or [] if line.startswith("+CPIN:")), None)
        sim_state = "no_answer" if v is None else CPIN.get(v, "other")  # no answer: an error such as SIM not inserted
        ready = card and sim_state == "ready"
        if ready and not mcc_mnc:
          imsi = next((line.strip() for line in self._at_send("AT+CIMI") or [] if not line.startswith("+")), "")
          mcc_mnc = imsi[:6] if imsi.isascii() and imsi.isdigit() and len(imsi) >= 6 else ""
        return {"sim_state": sim_state, "euicc": self._probe_euicc() if ready else None, "sim_home_mcc_mnc": mcc_mnc[:5]}
    except Exception:
      logging.exception("modem_status (probe) failed")
      return None

  def _probe_euicc(self) -> bool:
    """The probe of LPA.is_euicc (the caller holds AT_LOCK): only an eUICC opens a logical channel to the ISD-R applet.
    A channel the modem opened is closed again on every path (one retry), so it is not lost to the LPA."""
    ch = ""
    try:
      for line in self._at_send(f'AT+CCHO="{ISDR_AID}"') or []:
        n = line.split(":", 1)[1].strip() if line.startswith("+CCHO:") else ""
        if n.isascii() and n.isdecimal():
          ch = n
          return True
      return False
    finally:
      if ch and self._at_send(f"AT+CCHC={ch}") is None:
        self._at_send(f"AT+CCHC={ch}")

  @staticmethod
  def _parse_act(v: str) -> str:
    """Technology from a +CREG / +CGREG read reply (<n>,<stat>,<lac>,<ci>,<AcT>): only the unquoted AcT number, mapped
    through NETWORK_TYPE; the quoted LAC / TAC and cell id fields are skipped."""
    p = v.split(",")
    act = p[4].strip() if len(p) >= 5 else ""
    return NETWORK_TYPE.get(int(act), "unknown") if act.isascii() and act.isdecimal() else "unknown"

  def _log_registration(self, reg: str, greg: str, v: str, gv: str):
    """On each change of the registration pair (network, packet data) the daemon reads in SEARCHING."""
    try:
      if (reg, greg) == (self._reg_status["registration"], self._reg_status["data_registration"]):
        return
      technology = self._parse_act(v)
      if technology == "unknown":
        technology = self._parse_act(gv)
      self._reg_status = {"registration": reg, "data_registration": greg, "technology": technology}
      if self._probe_pending:
        self._update_probe()
      self._emit_status("registration", roaming_allowed=self._roaming_allowed)
    except Exception:
      logging.exception("modem_status (registration) failed")

  def _log_onroad(self):
    """Repeat the last status once per drive, ONROAD_REPEAT_DELAY after going onroad: loggerd records only onroad, so the
    events of a boot can predate the route. IsOffroad is the param manager keeps for processes without deviceState."""
    try:
      if self._read_param("IsOffroad") != "0":
        self._onroad_since, self._onroad_logged = None, False
        return
      now = time.monotonic()
      if self._onroad_since is None:
        self._onroad_since = now
      if not self._onroad_logged and self._sim_status and now - self._onroad_since >= ONROAD_REPEAT_DELAY:
        self._onroad_logged = True
        if self._probe_pending:
          self._update_probe()
        self._emit_status("onroad", roaming_allowed=self._roaming_allowed)
    except Exception:
      logging.exception("modem_status (onroad) failed")

  def _emit_status(self, reason: str, **extra):
    cloudlog.event("modem_status", reason=reason, **self._sim_status, **self._reg_status, **extra, error=True)

  def _do_searching(self):
    new_roaming = self._is_roaming_allowed()
    if new_roaming != self._roaming_allowed:
      logging.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      self._roaming_allowed = new_roaming

    v = self._atv("AT+CREG?", "+CREG:")
    if not v:
      return self._searching_idle()

    reg = self._parse_reg(v)
    gv = self._atv("AT+CGREG?", "+CGREG:") or ""
    greg = self._parse_reg(gv)
    self._log_registration(reg, greg, v, gv)
    logging.debug(f"creg={reg} cgreg={greg} roaming_allowed={self._roaming_allowed}")

    if reg == "roaming" and not self._roaming_allowed:
      self._publish_state(registration=reg)
      return State.SEARCHING

    if reg in ("home", "roaming") and greg in ("home", "roaming"):
      self._publish_state(registration=reg)
      return State.CONNECTING

    if reg != self.S.get("registration"):
      self._publish_state(registration=reg)
    return self._searching_idle()

  def _searching_idle(self):
    if self._sim_change or not os.path.exists(AT_PORT):
      logging.info(f"-> reconnecting (sim_change={self._sim_change} port={os.path.exists(AT_PORT)})")
      return State.DISCONNECTING
    return State.SEARCHING

  def _do_connecting(self):
    logging.info("starting pppd")
    self._ppp.reset_fail_counter()
    self._sim_change = False
    self._ppp.start()
    return State.CONNECTED

  def _handle_pppd_exit(self):
    if self._sim_change or not os.path.exists(AT_PORT):
      return State.DISCONNECTING
    give_up = self._ppp.record_fail()
    if give_up:
      logging.warning(f"PPP fail {self._ppp.fails}/{self._ppp.MAX_FAILS}, reconnecting")
      return State.DISCONNECTING
    logging.warning(f"PPP fail {self._ppp.fails}/{self._ppp.MAX_FAILS}, retrying")
    self._ppp.reset_data_port()
    if not os.path.exists(AT_PORT):
      return State.DISCONNECTING
    self._ppp.start()
    return State.CONNECTED

  def _params_changed(self) -> bool:
    new_apn = self._read_param("GsmApn")
    if new_apn != self._apn:
      logging.info(f"GsmApn changed: '{self._apn}' -> '{new_apn}'")
      return True
    new_roaming = self._is_roaming_allowed()
    if new_roaming != self._roaming_allowed:
      logging.info(f"roaming changed: {self._roaming_allowed} -> {new_roaming}")
      return True
    return False

  def _check_iccid(self, state):
    if state in (State.INITIALIZING, State.DISCONNECTING) or not self.S["iccid"]:
      return
    iccid = (self._atv("AT+QCCID", "+QCCID:") or "").rstrip("F")
    if iccid and iccid != self.S["iccid"]:
      logging.warning(f"iccid changed: {self.S['iccid']} -> {iccid}")
      self._sim_change = True

  def _do_connected(self):
    if self._ppp.has_exited():
      return self._handle_pppd_exit()

    if self._sim_change or not os.path.exists(AT_PORT) or self._params_changed():
      return State.DISCONNECTING

    self._poll()
    return State.CONNECTED

  def _do_disconnecting(self):
    logging.warning("reconnecting")
    self._publish_state(**INITIAL_STATE)
    self._ppp.kill()
    self._ppp.cleanup_routes()
    self._ppp.reset_data_port()
    self._sim_change = False
    return State.INITIALIZING

  def _poll_signal(self) -> dict:
    v = self._atv("AT+CSQ", "+CSQ:")
    if not v:
      return {}
    try:
      rssi = int(v.split(",")[0])
      if rssi == 99:
        return {}
      return {"signal_strength": rssi, "signal_quality": min(100, int(rssi / 31 * 100))}
    except (ValueError, IndexError):
      return {}

  def _poll_operator(self) -> dict:
    v = self._atv("AT+COPS?", "+COPS:")
    if not v:
      return {}
    p = v.split(",")
    out: dict = {}
    try:
      if len(p) >= 3:
        out["operator"] = p[2].strip('"')
      if len(p) >= 4:
        out["network_type"] = NETWORK_TYPE.get(int(p[3]), "unknown")
    except (ValueError, IndexError):
      pass
    return out

  def _poll_band(self) -> dict:
    v = self._atv("AT+QNWINFO", "+QNWINFO:")
    if not v:
      return {}
    info = v.replace('"', '').split(",")
    try:
      if len(info) >= 4:
        return {"band": info[2], "channel": int(info[3])}
    except ValueError:
      pass
    return {}

  def _poll_extra(self) -> dict:
    v = self._atv('AT+QENG="servingcell"', "+QENG:")
    return {"extra": v.replace('"', '')} if v else {}

  def _poll_temps(self) -> dict:
    v = self._atv("AT+QTEMP", "+QTEMP:")
    if not v:
      return {}
    try:
      return {"temperatures": [t for t in (int(x) for x in v.split(",") if x.strip()) if t != 255]}
    except (ValueError, IndexError):
      return {}

  def _poll_iface(self) -> dict:
    try:
      r = subprocess.run(["ip", "-4", "addr", "show", "ppp0"], capture_output=True, text=True, timeout=2)
      ip, peer = "", ""
      for line in r.stdout.splitlines():
        # `inet 10.x.x.x peer 10.64.64.64/32 ...`
        parts = line.strip().split()
        if "inet" in parts:
          i = parts.index("inet")
          ip = parts[i + 1].split("/")[0]
          if "peer" in parts:
            peer = parts[parts.index("peer") + 1].split("/")[0]
          break
      if ip:
        if self._ppp.maybe_install_routes(ip, peer):
          self._ppp.maybe_install_dns(self._read_cellular_dns())
        return {"ip_address": ip, "connected": True}
      if self.S["connected"]:
        return {"connected": False, "ip_address": ""}
    except Exception:
      pass
    return {}

  def _read_cellular_dns(self) -> list[str]:
    v = self._atv(f"AT+CGCONTRDP={DIAL_CID}", "+CGCONTRDP:")
    if not v:
      return []
    # +CGCONTRDP: <cid>,<bearer_id>,<apn>,<local_addr>,<gw_addr>,<dns_prim>,<dns_sec>,...
    fields = [f.strip().strip('"') for f in v.split(",")]
    dns_servers = []
    for d in fields[5:7]:
      try:
        dns_servers.append(str(IPv4Address(d)))
      except (AddressValueError, ValueError):
        pass
    if not dns_servers:
      dns_servers = [
        "8.8.8.8",  # Google
        "1.1.1.1",  # Cloudflare
      ]
      logging.warning(f"no cellular DNS servers reported by modem: {v!r}; using fallback {dns_servers}")
    return dns_servers

  def _poll_byte_counters(self) -> dict:
    try:
      with open("/sys/class/net/ppp0/statistics/tx_bytes") as f:
        tx = int(f.read().strip())
      with open("/sys/class/net/ppp0/statistics/rx_bytes") as f:
        rx = int(f.read().strip())
    except Exception:
      return {}
    return {"tx_bytes": tx, "rx_bytes": rx}

  def _poll(self):
    s: dict = {}
    for fn in (self._poll_signal, self._poll_operator, self._poll_band,
               self._poll_extra, self._poll_temps, self._poll_iface,
               self._poll_byte_counters):
      s.update(fn())
    if s:
      self._publish_state(**s)

  def run(self):
    logging.info("starting")
    self._publish_state(state=State.INITIALIZING.value)
    if self._has_modem_manager():
      subprocess.run(["sudo", "systemctl", "mask", "--runtime", "ModemManager"], capture_output=True)
      subprocess.run(["sudo", "systemctl", "stop", "ModemManager"], capture_output=True)
    self._ppp.kill()

    state = State.INITIALIZING

    handlers = {
      State.INITIALIZING: self._do_initializing,
      State.SEARCHING: self._do_searching,
      State.CONNECTING: self._do_connecting,
      State.CONNECTED: self._do_connected,
      State.DISCONNECTING: self._do_disconnecting,
    }

    while self.running:
      try:
        self._check_iccid(state)
        self._log_onroad()
        prev = state
        state = handlers[state]()
        if state != prev:
          self._publish_state(state=state.value)
          logging.info(f"{prev.value} -> {state.value}")
      except Exception:
        logging.exception(f"error in {state.value}")
        state = State.DISCONNECTING
      time.sleep(STATE_WAIT)

  def stop(self):
    self.running = False
    self._ppp.kill()
    self._ppp.cleanup_routes()
    try:
      os.remove(STATE_PATH)
    except FileNotFoundError:
      pass
    if self._has_modem_manager():
      subprocess.run(["sudo", "systemctl", "unmask", "--runtime", "ModemManager"], capture_output=True)
      subprocess.run(["sudo", "systemctl", "start", "ModemManager"], capture_output=True)


def main():
  m = Modem()

  def _sig(*_):
    m.running = False

  signal.signal(signal.SIGINT, _sig)
  signal.signal(signal.SIGTERM, _sig)
  m.run()
  m.stop()


if __name__ == "__main__":
  main()
