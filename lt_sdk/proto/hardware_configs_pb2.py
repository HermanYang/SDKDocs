# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lt_sdk/proto/hardware_configs.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='lt_sdk/proto/hardware_configs.proto',
  package='light',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n#lt_sdk/proto/hardware_configs.proto\x12\x05light*4\n\x0eHardwareConfig\x12\t\n\x05\x44\x45LTA\x10\x00\x12\t\n\x05\x42RAVO\x10\x01\x12\x0c\n\x08VANGUARD\x10\x02\x62\x06proto3')
)

_HARDWARECONFIG = _descriptor.EnumDescriptor(
  name='HardwareConfig',
  full_name='light.HardwareConfig',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DELTA', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BRAVO', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VANGUARD', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=46,
  serialized_end=98,
)
_sym_db.RegisterEnumDescriptor(_HARDWARECONFIG)

HardwareConfig = enum_type_wrapper.EnumTypeWrapper(_HARDWARECONFIG)
DELTA = 0
BRAVO = 1
VANGUARD = 2


DESCRIPTOR.enum_types_by_name['HardwareConfig'] = _HARDWARECONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


# @@protoc_insertion_point(module_scope)
