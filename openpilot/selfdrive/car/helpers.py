import capnp
from typing import Any

from openpilot.cereal import custom
from opendbc.car import structs

_FIELDS = '__dataclass_fields__'  # copy of dataclasses._FIELDS


def is_dataclass(obj):
  """Similar to dataclasses.is_dataclass without instance type check checking"""
  return hasattr(obj, _FIELDS)


def _asdictref_inner(obj) -> dict[str, Any] | Any:
  if is_dataclass(obj):
    ret = {}
    for field in getattr(obj, _FIELDS):  # similar to dataclasses.fields()
      ret[field] = _asdictref_inner(getattr(obj, field))
    return ret
  elif isinstance(obj, (tuple, list)):
    return type(obj)(_asdictref_inner(v) for v in obj)
  else:
    return obj


def asdictref(obj) -> dict[str, Any]:
  """
  Similar to dataclasses.asdict without recursive type checking and copy.deepcopy
  Note that the resulting dict will contain references to the original struct as a result
  """
  if not is_dataclass(obj):
    raise TypeError("asdictref() should be called on dataclass instances")

  return _asdictref_inner(obj)


def convert_to_capnp(struct: structs.CarParamsSP | structs.CarStateSP) -> capnp.lib.capnp._DynamicStructBuilder:
  struct_dict = asdictref(struct)

  if isinstance(struct, structs.CarParamsSP):
    struct_capnp = custom.CarParamsSP.new_message(**struct_dict)
  elif isinstance(struct, structs.CarStateSP):
    struct_capnp = custom.CarStateSP.new_message(**struct_dict)
  else:
    raise ValueError(f"Unsupported struct type: {type(struct)}")

  return struct_capnp


def _convert_lead_data(struct: capnp.lib.capnp._DynamicStructReader, field: str) -> structs.LeadData:
  if not struct._has(field):
    return structs.LeadData()
  lead = getattr(struct, field)
  return structs.LeadData(dRel=lead.dRel, yRel=lead.yRel, vRel=lead.vRel, aRel=lead.aRel, vLead=lead.vLead, dPath=lead.dPath,
                          vLat=lead.vLat, vLeadK=lead.vLeadK, aLeadK=lead.aLeadK, fcw=lead.fcw, status=lead.status,
                          aLeadTau=lead.aLeadTau, modelProb=lead.modelProb, radar=lead.radar, radarTrackId=lead.radarTrackId)


def convert_carControlSP(struct: capnp.lib.capnp._DynamicStructReader) -> structs.CarControlSP:
  """
  Copies a carControlSP reader into structs.CarControlSP, field by field (card runs this every cycle).
  The result equals building the dataclass from struct.to_dict(): a sub-struct that is not set keeps the dataclass
  defaults, enums become their names (str), DEPRECATED fields keep their defaults and params stays a list of dicts.
  test_car_helpers.py compares it with that conversion over the whole schema and fails on a field not copied here.
  """
  if struct._has('mads'):
    mads = struct.mads
    mads_dataclass = structs.ModularAssistiveDrivingSystem(state=str(mads.state), enabled=mads.enabled, active=mads.active,
                                                           available=mads.available)
  else:
    mads_dataclass = structs.ModularAssistiveDrivingSystem()

  if struct._has('intelligentCruiseButtonManagement'):
    icbm = struct.intelligentCruiseButtonManagement
    icbm_dataclass = structs.IntelligentCruiseButtonManagement(state=str(icbm.state), sendButton=str(icbm.sendButton), vTarget=icbm.vTarget)
  else:
    icbm_dataclass = structs.IntelligentCruiseButtonManagement()

  # TODO: convert params to structs.CarControlSP.Param once something reads them
  return structs.CarControlSP(mads=mads_dataclass, params=[p.to_dict() for p in struct.params],
                              leadOne=_convert_lead_data(struct, 'leadOne'), leadTwo=_convert_lead_data(struct, 'leadTwo'),
                              intelligentCruiseButtonManagement=icbm_dataclass)
