"""modem_status cloudlog event of the modem daemon: content per SIM case, no identity numbers or location, when it fires.

Runs without a modem: the AT channel is a scripted stub (Modem._at_send), the AT lock is the daemon's own flock on a file in
the test's tmp folder, and the event is captured as the exact string the swaglog IPC handler sends to logmessaged
(SwagFormatter), i.e. what becomes logMessage / errorLogMessage in rlogs and qlogs.
"""
import ast
import fcntl
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import openpilot.common.hardware.comma.modem as modem
from openpilot.common.logging_extra import SwagFormatter
from openpilot.common.swaglog import cloudlog

# test identities (made up)
IMEI = "490154203237518"
ICCID_OTHER = "8941017356204918273"      # a Swiss-style card id
ICCID_COMMA = "89852359876001122334"     # starts with comma's (Webbing) prefix 8985235
IMSI_OTHER = "228017766554433"           # home network 228 01
IMSI_COMMA = "234507788990011"           # home network 234 50
VERSION = "EG25GGBR07A08M2G"
# location fields of the +CREG / +CGREG replies (hex as sent by the modem, plus their decimal values)
LAC, CI, G_LAC, G_CI, E_TAC, E_CI = "1A2B", "01C2D3E", "3C4D", "0ABCDEF", "5E6F", "0FEDCBA"
LOCATION = ([LAC, CI, G_LAC, G_CI, E_TAC, E_CI] + [x.lstrip("0") for x in (CI, G_CI, E_CI)] +
            [str(int(x, 16)) for x in (LAC, CI, G_LAC, G_CI, E_TAC, E_CI)])
# unsolicited reports the modem can send on the AT port at any time (AT_INIT sets AT+CREG=2 / AT+CGREG=2, _configure_modem
# turns on +QSIMSTAT): the location fields are those above
URCS = [f'+CREG: 1,"{LAC}","{CI}",7', f'+CGREG: 1,"{G_LAC}","{G_CI}",7', f'+CEREG: 1,"{E_TAC}","{E_CI}",7',
        "+QSIMSTAT: 1,1", "+CPIN: READY"]

FIELDS_IDENTITY = {"event", "reason", "error", "modem_version", "sim_present", "sim_state", "euicc", "comma_profile",
                   "sim_home_mcc_mnc", "registration", "data_registration", "technology"}
FIELDS_REGISTRATION = FIELDS_IDENTITY | {"roaming_allowed"}


def creg(stat, act=7, lac=LAC, ci=CI, prefix="+CREG"):
  return f'{prefix}: 2,{stat},"{lac}","{ci}",{act}' if stat in (1, 5) else f"{prefix}: 2,{stat}"


def cgreg(stat, act=7):
  return creg(stat, act, G_LAC, G_CI, "+CGREG")


class Holder:
  """Another process's hold of AT_LOCK (the LPA, qcomgpsd). flock locks belong to the open file, so a second open() of the
  lock file conflicts with the daemon's hold also within one process."""

  def __init__(self):
    self.fd = None

  def take(self) -> bool:
    fd = os.open(modem.AT_LOCK, os.O_CREAT | os.O_RDWR)
    try:
      fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      os.close(fd)
      return False
    self.fd = fd
    return True

  def release(self):
    if self.fd is not None:
      fcntl.flock(self.fd, fcntl.LOCK_UN)
      os.close(self.fd)
      self.fd = None


def lock_held() -> bool:
  h = Holder()
  if h.take():
    h.release()
    return False
  return True


FAKES = []


class FakeModem(modem.Modem):
  """Modem with a scripted AT channel (an answer of None is an ERROR reply); no pppd, no params. The real _at and _at_lock
  run on a lock file in the test's tmp folder. open_channels is the modem's side of AT+CCHO / AT+CCHC."""

  def __init__(self, at_port, imei=IMEI, iccid=None, imsi=None, version=VERSION, cpin="READY", euicc=False, params=None, urcs=()):
    super().__init__()
    self.answers = {"AT+CGMI": ["Quectel"], "AT+CGSN": [imei] if imei else None, "AT+GMR": [version] if version else None}
    self.set_sim(iccid, imsi, cpin, euicc)
    self.params = params if params is not None else {"GsmRoaming": "1"}
    self.urcs = list(urcs)          # sent before every reply
    self.fail = {}                  # command -> how many more times it answers ERROR
    self.sent, self.held_at_send = [], []
    self.open_channels: set[str] = set()
    self.after_at = self.on_send = None
    self._ppp = type("NoPPP", (), {"kill": lambda s: None, "cleanup_routes": lambda s: None, "reset_data_port": lambda s: None})()
    self.at_port = at_port
    FAKES.append(self)

  def set_sim(self, iccid, imsi=None, cpin="READY", euicc=False):
    self.answers["AT+QCCID"] = [f"+QCCID: {iccid}F"] if iccid else None
    self.answers["AT+CIMI"] = [imsi] if imsi else None
    self.answers["AT+CPIN?"] = [f"+CPIN: {cpin}"] if (iccid and cpin) else None
    self.answers[f'AT+CCHO="{modem.ISDR_AID}"'] = ["+CCHO: 1"] if euicc else None

  def set_registration(self, stat, gstat=None, act=7):
    self.answers["AT+CREG?"] = [creg(stat, act)]
    self.answers["AT+CGREG?"] = [cgreg(stat if gstat is None else gstat, act)]

  def _at(self, cmd):
    r = super()._at(cmd)
    if self.after_at:
      self.after_at(cmd)
    return r

  def _at_send(self, cmd):
    self.sent.append(cmd)
    self.held_at_send.append(lock_held())
    if self.on_send:
      self.on_send(cmd)
    if self.fail.get(cmd):
      self.fail[cmd] -= 1
      return None
    answer = self.answers.get(cmd, [])
    if answer is None:
      return None
    if cmd.startswith("AT+CCHO="):
      self.open_channels.add(answer[0].split(":")[1].strip())
    if cmd.startswith("AT+CCHC="):
      self.open_channels.discard(cmd.split("=")[1])
    return self.urcs + list(answer)

  def _read_param(self, key):
    return self.params.get(key, "")

  def _publish_state(self, **kwargs):
    self.S.update(kwargs)


@pytest.fixture
def at_port(tmp_path, monkeypatch):
  port = tmp_path / "modem_at0"
  port.touch()
  monkeypatch.setattr(modem, "AT_PORT", str(port))
  monkeypatch.setattr(modem, "AT_LOCK", str(tmp_path / "modem.lock"))
  FAKES.clear()
  yield port
  for m in FAKES:  # every test: no logical channel left open on the modem, and the daemon holds no lock afterwards
    assert m.open_channels == set(), "logical channel left open"
  assert not lock_held()


@pytest.fixture
def records(monkeypatch):
  """Everything cloudlog emits during the test, as (level, string sent to logmessaged)."""
  out = []
  fmt = SwagFormatter(cloudlog)

  class Capture(logging.Handler):
    def emit(self, record):
      out.append((record.levelno, fmt.format(record)))

  monkeypatch.setattr(cloudlog, "handlers", [Capture()])
  return out


def events(records):
  msgs = [json.loads(r)["msg"] for _, r in records]
  return [m for m in msgs if isinstance(m, dict) and m.get("event") == "modem_status"]


def assert_no_identity(records):
  """String search: no IMEI, card id, IMSI past the 5-digit network code, or LAC / cell id in what cloudlog emitted."""
  # whole records (event plus swaglog context and metadata): the full numbers (short values could match timestamps or ids)
  for _, r in records:
    for s in (IMEI, ICCID_OTHER, ICCID_COMMA, IMSI_OTHER, IMSI_COMMA, IMSI_OTHER[:6], IMSI_COMMA[:6], CI, G_CI, E_CI):
      assert s.lower() not in r.lower(), f"{s!r} found"
  # the event itself: no location value in any form, no 4 consecutive digits of the IMEI or card id, no 6 of the IMSI
  for e in events(records):
    text = json.dumps(e)
    for s in LOCATION:
      assert s.lower() not in text.lower(), f"location value {s!r} found"
    for secret, n in ((IMEI, 4), (ICCID_OTHER, 4), (ICCID_COMMA, 4), (IMSI_OTHER, 6), (IMSI_COMMA, 6)):
      for i in range(len(secret) - n + 1):
        assert secret[i:i + n] not in text, f"part of an identity number found: {secret[i:i + n]!r}"


def test_test_values_do_not_collide():
  """Precondition of the string search: the values the event may hold share no 4-digit run with the identity numbers."""
  allowed = json.dumps([VERSION, IMSI_OTHER[:5], IMSI_COMMA[:5], "lte", "utran"])
  for secret in (IMEI, ICCID_OTHER, ICCID_COMMA):
    assert not any(secret[i:i + 4] in allowed for i in range(len(secret) - 3))
  assert not any(s.lower() in allowed.lower() for s in LOCATION)


def test_sim_absent(at_port, records):
  m = FakeModem(at_port, iccid=None)
  for _ in range(5):  # the daemon retries the identity read every loop while no card answers
    assert m._do_initializing() == modem.State.INITIALIZING
  ev = events(records)
  assert len(ev) == 1
  e = ev[0]
  assert set(e) == FIELDS_IDENTITY
  assert e["reason"] == "identity" and e["modem_version"] == VERSION
  assert e["sim_present"] is False and e["sim_state"] == "no_answer" and e["euicc"] is None and e["comma_profile"] is None
  assert e["sim_home_mcc_mnc"] == "" and e["registration"] == "unknown"
  assert not any(c.startswith("AT+CCHO") for c in m.sent)  # no eUICC probe without a card
  assert_no_identity(records)


def test_comma_card(at_port, records):
  m = FakeModem(at_port, iccid=ICCID_COMMA, imsi=IMSI_COMMA, euicc=True)
  assert m._do_initializing() == modem.State.SEARCHING
  (lvl, _), = records
  assert lvl == logging.ERROR  # error level: logmessaged also publishes it as errorLogMessage, which qlogs keep
  e, = events(records)
  assert set(e) == FIELDS_IDENTITY
  assert (e["sim_present"], e["sim_state"], e["euicc"], e["comma_profile"], e["sim_home_mcc_mnc"]) == (True, "ready", True, True, "23450")
  assert "AT+CCHC=1" in m.sent  # the probe closes the channel it opened
  assert_no_identity(records)


def test_other_card_home(at_port, records):
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, params={"GsmRoaming": "0"})
  assert m._do_initializing() == modem.State.SEARCHING
  m.set_registration(2, 2)
  for _ in range(3):
    assert m._do_searching() == modem.State.SEARCHING
  m.set_registration(1)
  assert m._do_searching() == modem.State.CONNECTING
  ident, searching, home = events(records)
  assert (ident["sim_present"], ident["euicc"], ident["comma_profile"], ident["sim_home_mcc_mnc"]) == (True, False, False, "22801")
  assert set(searching) == set(home) == FIELDS_REGISTRATION
  assert (searching["registration"], searching["data_registration"], searching["technology"]) == ("searching", "searching", "unknown")
  assert (home["reason"], home["registration"], home["data_registration"], home["technology"]) == ("registration", "home", "home", "lte")
  assert home["roaming_allowed"] is False and home["sim_home_mcc_mnc"] == "22801"  # SIM fields repeated in every event
  assert_no_identity(records)


@pytest.mark.parametrize("allowed", ["1", "0"])
def test_roaming(at_port, records, allowed):
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, params={"GsmRoaming": allowed})
  m._do_initializing()
  m.set_registration(5, act=9)
  states = [m._do_searching() for _ in range(4)]
  assert states[0] == (modem.State.CONNECTING if allowed == "1" else modem.State.SEARCHING)
  _, e = events(records)  # one event, not one per poll while roaming is refused
  assert (e["registration"], e["data_registration"], e["technology"], e["roaming_allowed"]) == ("roaming", "roaming", "lte", allowed == "1")
  assert_no_identity(records)


def test_pin_locked_card(at_port, records):
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=None, cpin="SIM PIN", euicc=True)  # the IMSI is unreadable until the PIN is entered
  m._do_initializing()
  e, = events(records)
  assert (e["sim_present"], e["sim_state"], e["euicc"], e["sim_home_mcc_mnc"]) == (True, "pin_required", None, "")
  assert not any(c.startswith("AT+CCHO") for c in m.sent)  # eUICC probed only for a ready card
  assert_no_identity(records)


def test_registration_changes_fire_once_each(at_port, records):
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER)
  m._do_initializing()
  for stat, polls in ((2, 3), (3, 2), (0, 2), (2, 1)):
    m.set_registration(stat)
    for _ in range(polls):
      m._do_searching()
  m.set_registration(1, gstat=2)   # network registered, packet data not yet: a change of the pair
  m._do_searching()
  m.set_registration(1)
  m._do_searching()
  regs = [(e["registration"], e["data_registration"]) for e in events(records)[1:]]
  assert regs == [("searching", "searching"), ("denied", "denied"), ("not_registered", "not_registered"),
                  ("searching", "searching"), ("home", "searching"), ("home", "home")]
  assert_no_identity(records)


def test_identity_event_per_new_identity(at_port, records):
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER)
  m._do_initializing()
  m.set_registration(2)
  m._do_searching()
  # reconnect with the same card: identity read again, no new identity event, registration unchanged -> no event
  m._do_disconnecting()
  m._do_initializing()
  m._do_searching()
  assert [e["reason"] for e in events(records)] == ["identity", "registration"]
  # card swapped: _check_iccid flags it, the daemon reconnects and reads the new identity -> one identity event
  m.set_sim(ICCID_COMMA, IMSI_COMMA, euicc=True)
  m._check_iccid(modem.State.SEARCHING)
  assert m._searching_idle() == modem.State.DISCONNECTING
  m._do_disconnecting()
  m._do_initializing()
  m._do_searching()
  ev = events(records)
  assert [e["reason"] for e in ev] == ["identity", "registration", "identity", "registration"]
  assert ev[2]["comma_profile"] is True and ev[2]["registration"] == "unknown"  # registration reset at a new identity
  assert_no_identity(records)


def test_onroad_repeat_once_per_drive(at_port, records, monkeypatch):
  """loggerd records only onroad: the last status is repeated once per drive, ONROAD_REPEAT_DELAY after going onroad."""
  clock = [100.0]
  monkeypatch.setattr(modem, "time", SimpleNamespace(monotonic=lambda: clock[0]))
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, params={"GsmRoaming": "1", "IsOffroad": "1"})
  m._log_onroad()  # offroad, no status yet
  m._do_initializing()
  m.set_registration(1)
  m._do_searching()
  m.params["IsOffroad"] = "0"   # drive 1 starts
  for _ in range(int(modem.ONROAD_REPEAT_DELAY)):  # one loop per second until just before the delay
    m._log_onroad()
    clock[0] += 1.0
  assert [e["reason"] for e in events(records)] == ["identity", "registration"]
  for _ in range(30):
    m._log_onroad()
    clock[0] += 1.0
  ev = events(records)
  assert [e["reason"] for e in ev] == ["identity", "registration", "onroad"]
  assert set(ev[2]) == FIELDS_REGISTRATION
  assert (ev[2]["sim_present"], ev[2]["sim_home_mcc_mnc"], ev[2]["registration"], ev[2]["technology"]) == (True, "22801", "home", "lte")
  m.params["IsOffroad"] = "1"   # drive ends, drive 2 starts
  m._log_onroad()
  m.params["IsOffroad"] = "0"
  for _ in range(30):
    m._log_onroad()
    clock[0] += 1.0
  assert [e["reason"] for e in events(records)] == ["identity", "registration", "onroad", "onroad"]
  assert_no_identity(records)


def test_run_loop(at_port, records, monkeypatch):
  """The daemon's own loop (1 s per state step): one identity, one registration and one onroad event in 60 loops."""
  clock = [0.0]
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, params={"GsmRoaming": "1", "IsOffroad": "0"})
  m.set_registration(2)

  def sleep(dt):
    clock[0] += dt
    m.running = clock[0] < 60
  monkeypatch.setattr(modem, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
  monkeypatch.setattr(modem.Modem, "_has_modem_manager", staticmethod(lambda: False))
  m.run()
  assert [e["reason"] for e in events(records)] == ["identity", "registration", "onroad"]
  assert m.sent.count("AT+CPIN?") == 1 and m.sent.count("AT+CREG?") > 50  # status read once, registration polled every loop
  assert_no_identity(records)


def test_onroad_waits_for_the_first_identity(at_port, records, monkeypatch):
  clock = [0.0]
  monkeypatch.setattr(modem, "time", SimpleNamespace(monotonic=lambda: clock[0]))
  m = FakeModem(at_port, imei="", iccid=None, version="", params={"IsOffroad": "0"})  # modem not answering yet
  for _ in range(30):
    m._log_onroad()
    clock[0] += 1.0
  assert events(records) == []
  m.answers.update({"AT+CGSN": [IMEI], "AT+GMR": [VERSION]})
  m.set_sim(ICCID_OTHER, IMSI_OTHER)
  m._do_initializing()
  m._log_onroad()
  assert [e["reason"] for e in events(records)] == ["identity", "onroad"]


def test_no_event_when_modem_silent(at_port, records):
  m = FakeModem(at_port, imei="", iccid=None, version="")  # e.g. the LPA holds the AT port: everything reads empty
  m._do_initializing()
  assert events(records) == []


@pytest.mark.parametrize("version", [f"EG25 {IMEI} \x00<x>", f"EG25{IMEI}", "EG25GGBR07A08M2G <rev>", "EG25G12345678R",
                                     URCS[0], f'1,"{LAC}","{CI}",7', f"EG25:{LAC}{CI}", "_EG25", "E" * 41])
def test_version_logged_only_as_a_plain_version_string(at_port, records, version):
  """No character stripping: a line that is not a plain version string as a whole is logged as "" (R1)."""
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, version=version)
  m._do_initializing()
  e, = events(records)
  assert e["modem_version"] == ""
  assert_no_identity(records)


def test_version_kept_when_plain(at_port, records):
  for v in (VERSION, "EG25GGBR07A08M2G_01.002.01.002", "E" * 40):
    records.clear()
    FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, version=v)._do_initializing()
    assert events(records)[0]["modem_version"] == v


def test_rejected_versions_do_not_repeat_the_identity_event(at_port, records):
  """The validated version is what counts as "the same identity": two different rejected lines give one event."""
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, version=f'+CGREG: 1,"{LAC}","{CI}",7 EG25')
  m._do_initializing()
  m.answers["AT+GMR"] = [f'EG25 "{G_LAC}","{G_CI}"']
  m._do_disconnecting()
  m._do_initializing()
  e, = events(records)
  assert e["modem_version"] == ""
  assert_no_identity(records)


def test_unsolicited_reports_before_every_reply(at_port, records):
  """+CREG / +CGREG / +CEREG reports with LAC / TAC and cell id, +QSIMSTAT and +CPIN before the reply of every command the
  event uses: no location value anywhere, the right values, and no extra identity event when the reports come and go (R1)."""
  m = FakeModem(at_port, iccid=ICCID_COMMA, imsi=IMSI_COMMA, euicc=True, urcs=URCS)
  assert m._do_initializing() == modem.State.SEARCHING
  for stat in (2, 1, 5):
    m.set_registration(stat)
    m._do_searching()
  m._do_disconnecting()
  m.urcs = []
  m._do_initializing()           # same card, now without reports: no new identity event
  m.urcs = URCS[::-1]
  m._do_disconnecting()
  m._do_initializing()
  ident = [e for e in events(records) if e["reason"] == "identity"]
  assert len(ident) == 1
  e = ident[0]
  assert (e["modem_version"], e["sim_present"], e["sim_state"], e["euicc"], e["comma_profile"], e["sim_home_mcc_mnc"]) == \
         (VERSION, True, "ready", True, True, "23450")
  for c in ("AT+CGSN", "AT+QCCID", "AT+CIMI", "AT+GMR", "AT+CPIN?", "AT+CREG?", "AT+CGREG?", f'AT+CCHO="{modem.ISDR_AID}"'):
    assert c in m.sent
  for _, r in records:
    for s in LOCATION:
      assert s.lower() not in json.dumps(json.loads(r)["msg"]).lower(), f"location value {s!r} found"
  assert_no_identity(records)


def test_at_keeps_its_behaviour(at_port):
  """_at = one non-blocking hold of AT_LOCK around _at_send: [] when another process holds the lock (nothing sent) or on an
  error reply, the reply lines otherwise."""
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER)
  h = Holder()
  assert h.take()
  assert m._at("AT+CGMI") == [] and m.sent == []
  h.release()
  assert m._at("AT+CGMI") == ["Quectel"] and m.held_at_send == [True]
  assert m._at("AT+CIMI") == [IMSI_OTHER]
  m.answers["AT+CIMI"] = None
  assert m._at("AT+CIMI") == []
  assert not lock_held()


def test_probe_holds_the_lock_across_ccho_and_cchc(at_port, records):
  """Another user tries to take AT_LOCK right after AT+CCHO (qcomgpsd at car start, the LPA): it cannot, the channel is
  closed under the same hold (R2)."""
  m = FakeModem(at_port, iccid=ICCID_COMMA, imsi=IMSI_COMMA, euicc=True)
  h = Holder()
  got_in = []
  m.on_send = lambda cmd: got_in.append(h.take()) if cmd.startswith("AT+CCHO") else None
  m._do_initializing()
  h.release()
  assert got_in == [False]
  probe = [(c, held) for c, held in zip(m.sent, m.held_at_send, strict=True) if c in ("AT+CPIN?", "AT+CCHC=1") or c.startswith("AT+CCHO")]
  assert [c for c, _ in probe] == ["AT+CPIN?", f'AT+CCHO="{modem.ISDR_AID}"', "AT+CCHC=1"] and all(held for _, held in probe)
  e, = events(records)
  assert (e["sim_state"], e["euicc"]) == ("ready", True)


def test_channel_closed_when_cchc_fails_once(at_port, records):
  m = FakeModem(at_port, iccid=ICCID_COMMA, imsi=IMSI_COMMA, euicc=True)
  m.fail["AT+CCHC=1"] = 1
  m._do_initializing()
  assert m.sent.count("AT+CCHC=1") == 2   # one retry; the fixture checks that the channel is closed
  assert events(records)[0]["euicc"] is True


def test_probe_exception_releases_lock_and_reports_unknown(at_port, records):
  """An unexpected exception inside the probe: lock released, nothing opened, fields "unknown", retried later."""
  m = FakeModem(at_port, iccid=ICCID_COMMA, imsi=IMSI_COMMA, euicc=True)

  def boom(cmd):
    if cmd == "AT+CPIN?" and m.sent.count(cmd) == 1:
      raise ValueError("unexpected")
  m.on_send = boom
  assert m._do_initializing() == modem.State.SEARCHING
  e, = events(records)
  assert (e["sim_state"], e["euicc"], e["sim_home_mcc_mnc"]) == ("unknown", None, "23450")
  m.set_registration(1)
  m._do_searching()
  assert (events(records)[1]["sim_state"], events(records)[1]["euicc"]) == ("ready", True)


def test_channel_closed_when_the_event_fails(at_port, records, monkeypatch):
  def boom(*a, **k):
    raise RuntimeError("cloudlog down")
  monkeypatch.setattr(modem.cloudlog, "event", boom)
  m = FakeModem(at_port, iccid=ICCID_COMMA, imsi=IMSI_COMMA, euicc=True)
  assert m._do_initializing() == modem.State.SEARCHING
  assert "AT+CCHC=1" in m.sent   # the fixture checks: channel closed, lock released


def test_lock_busy_reports_unknown_and_retries_at_the_next_trigger(at_port, records, monkeypatch):
  """AT_LOCK taken by another user after the identity read: the event says "unknown" (not "no_answer" / False), no wrong
  value is kept, and the probe runs again at the next trigger: registration change, then onroad (R2)."""
  clock = [0.0]
  monkeypatch.setattr(modem, "time", SimpleNamespace(monotonic=lambda: clock[0]))
  m = FakeModem(at_port, iccid=ICCID_COMMA, imsi=IMSI_COMMA, euicc=True, params={"GsmRoaming": "1", "IsOffroad": "1"})
  h = Holder()
  m.after_at = lambda cmd: h.take() if cmd == "AT+GMR" else None
  assert m._do_initializing() == modem.State.SEARCHING
  m.after_at = None
  e, = events(records)
  assert (e["sim_present"], e["sim_state"], e["euicc"], e["comma_profile"], e["sim_home_mcc_mnc"]) == (True, "unknown", None, True, "23450")
  assert "AT+CPIN?" not in m.sent and not any(c.startswith("AT+CCHO") for c in m.sent)
  # registration change while the lock is still held: the registration poll itself reads nothing, no event, still unknown
  m.set_registration(2)
  m._do_searching()
  assert len(events(records)) == 1
  h.release()
  m._do_searching()  # lock free: registration event, probe retried
  r = events(records)[1]
  assert (r["reason"], r["registration"], r["sim_state"], r["euicc"]) == ("registration", "searching", "ready", True)
  assert m.sent.count("AT+CPIN?") == 1
  m.set_registration(1)
  m._do_searching()
  assert m.sent.count("AT+CPIN?") == 1   # no further probes once it succeeded
  assert_no_identity(records)


def test_lock_busy_retried_at_onroad(at_port, records, monkeypatch):
  clock = [0.0]
  monkeypatch.setattr(modem, "time", SimpleNamespace(monotonic=lambda: clock[0]))
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER, params={"GsmRoaming": "1", "IsOffroad": "1"})
  h = Holder()
  m.after_at = lambda cmd: h.take() if cmd == "AT+GMR" else None
  m._do_initializing()
  m.after_at = None
  h.release()
  m.params["IsOffroad"] = "0"
  for _ in range(15):
    m._log_onroad()
    clock[0] += 1.0
  ident, onroad = events(records)
  assert (ident["sim_state"], ident["euicc"]) == ("unknown", None)
  assert (onroad["reason"], onroad["sim_state"], onroad["euicc"], onroad["sim_home_mcc_mnc"]) == ("onroad", "ready", False, "22801")


def test_lock_busy_without_card_retried_at_the_next_read(at_port, records):
  """No card, daemon in its identity retry loop: the lock is busy at the first read; the next read probes again and reports
  the real state in a second identity event, then no more events."""
  m = FakeModem(at_port, iccid=None)
  h = Holder()
  m.after_at = lambda cmd: h.take() if cmd == "AT+GMR" else None
  m._do_initializing()
  m.after_at = None
  h.release()
  for _ in range(4):
    m._do_initializing()
  ev = events(records)
  assert [(e["reason"], e["sim_present"], e["sim_state"]) for e in ev] == [("identity", False, "unknown"), ("identity", False, "no_answer")]
  assert m.sent.count("AT+CPIN?") == 1


def test_home_network_read_under_the_lock_when_the_identity_read_missed_it(at_port, records):
  """AT+CIMI of the identity read blocked by another lock holder: the probe reads it again under its own hold (R2)."""
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER)
  h = Holder()
  m.after_at = lambda cmd: h.take() if cmd == "AT+QCCID" else h.release() if cmd == "AT+CIMI" else None
  m._do_initializing()
  e, = events(records)
  assert e["sim_home_mcc_mnc"] == "22801" and m.sent.count("AT+CIMI") == 1
  assert_no_identity(records)


def test_location_fields_never_parsed():
  assert modem.Modem._parse_act(creg(1, 7)) == "lte"
  assert modem.Modem._parse_act(creg(1, 2)) == "utran"
  assert modem.Modem._parse_act("+CREG: 2,1") == "unknown"
  # a reply whose 5th field is quoted (a LAC / cell id in an unexpected place) yields no value
  assert modem.Modem._parse_act('2,1,"1A2B","01C2D3E","7"') == "unknown"
  assert modem.Modem._parse_act('1,"1A2B","01C2D3E",7') == "unknown"   # URC layout without <n>: no 5th field


def test_event_failure_never_breaks_the_daemon(at_port, records, monkeypatch):
  def boom(*a, **k):
    raise RuntimeError("cloudlog down")
  monkeypatch.setattr(modem.cloudlog, "event", boom)
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER)
  assert m._do_initializing() == modem.State.SEARCHING
  m.set_registration(1)
  assert m._do_searching() == modem.State.CONNECTING


def test_python_logging_stays_out_of_cloudlog(at_port, records, caplog):
  """The daemon's own identity line (with IMEI and card id) goes to Python logging only, never through cloudlog."""
  caplog.set_level(logging.INFO)
  m = FakeModem(at_port, iccid=ICCID_OTHER, imsi=IMSI_OTHER)
  m._do_initializing()
  assert any(IMEI in r.getMessage() for r in caplog.records)  # unchanged line, root logger
  assert cloudlog.propagate is True and cloudlog.parent is None  # swaglog's logger is not under the root logger
  assert_no_identity(records)


def test_isdr_aid_matches_lpa():
  src = (Path(modem.__file__).parents[2] / "esim" / "lpa.py").read_text()
  values = [n.value.value for n in ast.parse(src).body
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ISDR_AID" for t in n.targets)]
  assert values == [modem.ISDR_AID]
