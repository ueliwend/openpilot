import dataclasses
import math
import unittest

from openpilot.common.fuzzy import Fuzzy, capnp_random_dict, fuzzy_test
from openpilot.cereal import custom
from opendbc.car import structs
from openpilot.selfdrive.car.helpers import convert_carControlSP


def convert_carControlSP_generic(struct) -> structs.CarControlSP:
  """The generic to_dict() conversion that convert_carControlSP replaced, kept here as its reference."""
  def remove_deprecated(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.endswith('DEPRECATED')}

  struct_dict = struct.to_dict()
  struct_dataclass = structs.CarControlSP(**remove_deprecated(struct_dict))
  struct_dataclass.mads = structs.ModularAssistiveDrivingSystem(**remove_deprecated(struct_dict.get('mads', {})))
  struct_dataclass.leadOne = structs.LeadData(**remove_deprecated(struct_dict.get('leadOne', {})))
  struct_dataclass.leadTwo = structs.LeadData(**remove_deprecated(struct_dict.get('leadTwo', {})))
  struct_dataclass.intelligentCruiseButtonManagement = structs.IntelligentCruiseButtonManagement(
    **remove_deprecated(struct_dict.get('intelligentCruiseButtonManagement', {}))
  )
  return struct_dataclass


def assert_identical(test, a, b, path='CarControlSP'):
  """Same types and values everywhere; NaN equals NaN and -0.0 differs from 0.0."""
  test.assertIs(type(a), type(b), path)
  if dataclasses.is_dataclass(a):
    for f in dataclasses.fields(a):
      assert_identical(test, getattr(a, f.name), getattr(b, f.name), f'{path}.{f.name}')
  elif isinstance(a, list):
    test.assertEqual(len(a), len(b), path)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
      assert_identical(test, x, y, f'{path}[{i}]')
  elif isinstance(a, dict):
    test.assertEqual(a.keys(), b.keys(), path)
    for k in a:
      assert_identical(test, a[k], b[k], f'{path}.{k}')
  elif isinstance(a, float):
    test.assertTrue((math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1., a) == math.copysign(1., b)), f'{path}: {a} != {b}')
  else:
    test.assertEqual(a, b, path)


def non_default_dict(schema) -> dict:
  """Every field of the schema (DEPRECATED ones included) set to a value that differs from its default."""
  ret = {}
  for name, field in schema.fields.items():
    slot_type = field.proto.slot.type
    kind = slot_type.which()
    if kind == 'struct':
      ret[name] = non_default_dict(field.schema)
    elif kind == 'list':
      ret[name] = [non_default_dict(field.schema.elementType)]
    elif kind == 'enum':
      ret[name] = list(field.schema.enumerants)[-1]
    else:
      ret[name] = {'bool': True, 'text': 'x', 'data': b'x'}.get(kind, 7)
  return ret


class TestConvertCarControlSP(unittest.TestCase):
  def check(self, msg_dict):
    reader = custom.CarControlSP.new_message(**msg_dict).as_reader()
    assert_identical(self, convert_carControlSP(reader), convert_carControlSP_generic(reader))

  @fuzzy_test(max_examples=300)
  def test_random_messages(self, fuzzy: Fuzzy):
    # whole schema with NaN / inf, sub-structs randomly left unset, DEPRECATED fields randomly set
    msg = capnp_random_dict(fuzzy, custom.CarControlSP.schema)
    for name in list(msg):
      if fuzzy.boolean():
        del msg[name]
    for name in ('leadOne', 'leadTwo'):
      if name in msg and fuzzy.boolean():
        msg[name]['aLeadDEPRECATED'] = fuzzy.floating(32)
    for param in msg.get('params', []):
      if fuzzy.boolean():
        param['valueDEPRECATED'] = fuzzy.text(max_size=10)
    self.check(msg)

  def test_empty_message(self):
    self.check({})

  def test_every_field_is_copied(self):
    # each schema field on its own: equal to the generic conversion and different from an empty message,
    # so a field added to the schema but not to convert_carControlSP fails here
    default = convert_carControlSP(custom.CarControlSP.new_message().as_reader())
    full = non_default_dict(custom.CarControlSP.schema)

    def single_fields(d, path=()):
      for name, value in d.items():
        if isinstance(value, dict):
          yield from single_fields(value, path + (name,))
        else:
          yield path + (name,), value

    n_checked = 0
    for path, value in single_fields(full):
      msg = value
      for name in reversed(path):
        msg = {name: msg}
      with self.subTest(field='.'.join(path)):
        self.check(msg)
        if not path[-1].endswith('DEPRECATED'):
          reader = custom.CarControlSP.new_message(**msg).as_reader()
          self.assertNotEqual(convert_carControlSP(reader), default)
        n_checked += 1
    self.assertGreaterEqual(n_checked, 25)
