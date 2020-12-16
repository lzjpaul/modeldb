# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: modeldb/data/ModelData.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='modeldb/data/ModelData.proto',
  package='ai.verta.modeldb.data',
  syntax='proto3',
  serialized_options=b'P\001ZCgithub.com/VertaAI/modeldb/protos/gen/go/protos/public/modeldb/data',
  serialized_pb=b'\n\x1cmodeldb/data/ModelData.proto\x12\x15\x61i.verta.modeldb.data\x1a\x1cgoogle/api/annotations.proto\"Q\n\x11ModelDataMetadata\x12\x18\n\x10timestamp_millis\x18\x01 \x01(\x03\x12\x10\n\x08model_id\x18\x02 \x01(\t\x12\x10\n\x08\x65ndpoint\x18\x03 \x01(\t\"U\n\tModelData\x12:\n\x08metadata\x18\x01 \x01(\x0b\x32(.ai.verta.modeldb.data.ModelDataMetadata\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\t\"X\n\x15StoreModelDataRequest\x12\x33\n\tmodelData\x18\x01 \x01(\x0b\x32 .ai.verta.modeldb.data.ModelData\x1a\n\n\x08Response\"\x87\x01\n\x13GetModelDataRequest\x12\x19\n\x11start_time_millis\x18\x01 \x01(\x03\x12\x17\n\x0f\x65nd_time_millis\x18\x02 \x01(\x03\x12\x10\n\x08model_id\x18\x03 \x01(\t\x12\x10\n\x08\x65ndpoint\x18\x04 \x01(\t\x1a\x18\n\x08Response\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t\"\xa1\x01\n\x17GetModelDataDiffRequest\x12\x19\n\x11start_time_millis\x18\x01 \x01(\x03\x12\x17\n\x0f\x65nd_time_millis\x18\x02 \x01(\x03\x12\x12\n\nmodel_id_a\x18\x03 \x01(\t\x12\x12\n\nmodel_id_b\x18\x04 \x01(\t\x12\x10\n\x08\x65ndpoint\x18\x05 \x01(\t\x1a\x18\n\x08Response\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\t2\xe0\x03\n\x10ModelDataService\x12\x99\x01\n\x0eStoreModelData\x12,.ai.verta.modeldb.data.StoreModelDataRequest\x1a\x35.ai.verta.modeldb.data.StoreModelDataRequest.Response\"\"\x82\xd3\xe4\x93\x02\x1c\x1a\x17/v1/data/storeModelData:\x01*\x12\x8e\x01\n\x0cGetModelData\x12*.ai.verta.modeldb.data.GetModelDataRequest\x1a\x33.ai.verta.modeldb.data.GetModelDataRequest.Response\"\x1d\x82\xd3\xe4\x93\x02\x17\x12\x15/v1/data/getModelData\x12\x9e\x01\n\x10GetModelDataDiff\x12..ai.verta.modeldb.data.GetModelDataDiffRequest\x1a\x37.ai.verta.modeldb.data.GetModelDataDiffRequest.Response\"!\x82\xd3\xe4\x93\x02\x1b\x12\x19/v1/data/getModelDataDiffBGP\x01ZCgithub.com/VertaAI/modeldb/protos/gen/go/protos/public/modeldb/datab\x06proto3'
  ,
  dependencies=[google_dot_api_dot_annotations__pb2.DESCRIPTOR,])




_MODELDATAMETADATA = _descriptor.Descriptor(
  name='ModelDataMetadata',
  full_name='ai.verta.modeldb.data.ModelDataMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp_millis', full_name='ai.verta.modeldb.data.ModelDataMetadata.timestamp_millis', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_id', full_name='ai.verta.modeldb.data.ModelDataMetadata.model_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endpoint', full_name='ai.verta.modeldb.data.ModelDataMetadata.endpoint', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=166,
)


_MODELDATA = _descriptor.Descriptor(
  name='ModelData',
  full_name='ai.verta.modeldb.data.ModelData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='metadata', full_name='ai.verta.modeldb.data.ModelData.metadata', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='ai.verta.modeldb.data.ModelData.data', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=168,
  serialized_end=253,
)


_STOREMODELDATAREQUEST_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='ai.verta.modeldb.data.StoreModelDataRequest.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=333,
  serialized_end=343,
)

_STOREMODELDATAREQUEST = _descriptor.Descriptor(
  name='StoreModelDataRequest',
  full_name='ai.verta.modeldb.data.StoreModelDataRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='modelData', full_name='ai.verta.modeldb.data.StoreModelDataRequest.modelData', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_STOREMODELDATAREQUEST_RESPONSE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=255,
  serialized_end=343,
)


_GETMODELDATAREQUEST_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='ai.verta.modeldb.data.GetModelDataRequest.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='ai.verta.modeldb.data.GetModelDataRequest.Response.data', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=457,
  serialized_end=481,
)

_GETMODELDATAREQUEST = _descriptor.Descriptor(
  name='GetModelDataRequest',
  full_name='ai.verta.modeldb.data.GetModelDataRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='start_time_millis', full_name='ai.verta.modeldb.data.GetModelDataRequest.start_time_millis', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_time_millis', full_name='ai.verta.modeldb.data.GetModelDataRequest.end_time_millis', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_id', full_name='ai.verta.modeldb.data.GetModelDataRequest.model_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endpoint', full_name='ai.verta.modeldb.data.GetModelDataRequest.endpoint', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GETMODELDATAREQUEST_RESPONSE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=346,
  serialized_end=481,
)


_GETMODELDATADIFFREQUEST_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='ai.verta.modeldb.data.GetModelDataDiffRequest.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='ai.verta.modeldb.data.GetModelDataDiffRequest.Response.data', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=457,
  serialized_end=481,
)

_GETMODELDATADIFFREQUEST = _descriptor.Descriptor(
  name='GetModelDataDiffRequest',
  full_name='ai.verta.modeldb.data.GetModelDataDiffRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='start_time_millis', full_name='ai.verta.modeldb.data.GetModelDataDiffRequest.start_time_millis', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_time_millis', full_name='ai.verta.modeldb.data.GetModelDataDiffRequest.end_time_millis', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_id_a', full_name='ai.verta.modeldb.data.GetModelDataDiffRequest.model_id_a', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_id_b', full_name='ai.verta.modeldb.data.GetModelDataDiffRequest.model_id_b', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='endpoint', full_name='ai.verta.modeldb.data.GetModelDataDiffRequest.endpoint', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GETMODELDATADIFFREQUEST_RESPONSE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=484,
  serialized_end=645,
)

_MODELDATA.fields_by_name['metadata'].message_type = _MODELDATAMETADATA
_STOREMODELDATAREQUEST_RESPONSE.containing_type = _STOREMODELDATAREQUEST
_STOREMODELDATAREQUEST.fields_by_name['modelData'].message_type = _MODELDATA
_GETMODELDATAREQUEST_RESPONSE.containing_type = _GETMODELDATAREQUEST
_GETMODELDATADIFFREQUEST_RESPONSE.containing_type = _GETMODELDATADIFFREQUEST
DESCRIPTOR.message_types_by_name['ModelDataMetadata'] = _MODELDATAMETADATA
DESCRIPTOR.message_types_by_name['ModelData'] = _MODELDATA
DESCRIPTOR.message_types_by_name['StoreModelDataRequest'] = _STOREMODELDATAREQUEST
DESCRIPTOR.message_types_by_name['GetModelDataRequest'] = _GETMODELDATAREQUEST
DESCRIPTOR.message_types_by_name['GetModelDataDiffRequest'] = _GETMODELDATADIFFREQUEST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ModelDataMetadata = _reflection.GeneratedProtocolMessageType('ModelDataMetadata', (_message.Message,), {
  'DESCRIPTOR' : _MODELDATAMETADATA,
  '__module__' : 'modeldb.data.ModelData_pb2'
  # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.ModelDataMetadata)
  })
_sym_db.RegisterMessage(ModelDataMetadata)

ModelData = _reflection.GeneratedProtocolMessageType('ModelData', (_message.Message,), {
  'DESCRIPTOR' : _MODELDATA,
  '__module__' : 'modeldb.data.ModelData_pb2'
  # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.ModelData)
  })
_sym_db.RegisterMessage(ModelData)

StoreModelDataRequest = _reflection.GeneratedProtocolMessageType('StoreModelDataRequest', (_message.Message,), {

  'Response' : _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), {
    'DESCRIPTOR' : _STOREMODELDATAREQUEST_RESPONSE,
    '__module__' : 'modeldb.data.ModelData_pb2'
    # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.StoreModelDataRequest.Response)
    })
  ,
  'DESCRIPTOR' : _STOREMODELDATAREQUEST,
  '__module__' : 'modeldb.data.ModelData_pb2'
  # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.StoreModelDataRequest)
  })
_sym_db.RegisterMessage(StoreModelDataRequest)
_sym_db.RegisterMessage(StoreModelDataRequest.Response)

GetModelDataRequest = _reflection.GeneratedProtocolMessageType('GetModelDataRequest', (_message.Message,), {

  'Response' : _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), {
    'DESCRIPTOR' : _GETMODELDATAREQUEST_RESPONSE,
    '__module__' : 'modeldb.data.ModelData_pb2'
    # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.GetModelDataRequest.Response)
    })
  ,
  'DESCRIPTOR' : _GETMODELDATAREQUEST,
  '__module__' : 'modeldb.data.ModelData_pb2'
  # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.GetModelDataRequest)
  })
_sym_db.RegisterMessage(GetModelDataRequest)
_sym_db.RegisterMessage(GetModelDataRequest.Response)

GetModelDataDiffRequest = _reflection.GeneratedProtocolMessageType('GetModelDataDiffRequest', (_message.Message,), {

  'Response' : _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), {
    'DESCRIPTOR' : _GETMODELDATADIFFREQUEST_RESPONSE,
    '__module__' : 'modeldb.data.ModelData_pb2'
    # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.GetModelDataDiffRequest.Response)
    })
  ,
  'DESCRIPTOR' : _GETMODELDATADIFFREQUEST,
  '__module__' : 'modeldb.data.ModelData_pb2'
  # @@protoc_insertion_point(class_scope:ai.verta.modeldb.data.GetModelDataDiffRequest)
  })
_sym_db.RegisterMessage(GetModelDataDiffRequest)
_sym_db.RegisterMessage(GetModelDataDiffRequest.Response)


DESCRIPTOR._options = None

_MODELDATASERVICE = _descriptor.ServiceDescriptor(
  name='ModelDataService',
  full_name='ai.verta.modeldb.data.ModelDataService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=648,
  serialized_end=1128,
  methods=[
  _descriptor.MethodDescriptor(
    name='StoreModelData',
    full_name='ai.verta.modeldb.data.ModelDataService.StoreModelData',
    index=0,
    containing_service=None,
    input_type=_STOREMODELDATAREQUEST,
    output_type=_STOREMODELDATAREQUEST_RESPONSE,
    serialized_options=b'\202\323\344\223\002\034\032\027/v1/data/storeModelData:\001*',
  ),
  _descriptor.MethodDescriptor(
    name='GetModelData',
    full_name='ai.verta.modeldb.data.ModelDataService.GetModelData',
    index=1,
    containing_service=None,
    input_type=_GETMODELDATAREQUEST,
    output_type=_GETMODELDATAREQUEST_RESPONSE,
    serialized_options=b'\202\323\344\223\002\027\022\025/v1/data/getModelData',
  ),
  _descriptor.MethodDescriptor(
    name='GetModelDataDiff',
    full_name='ai.verta.modeldb.data.ModelDataService.GetModelDataDiff',
    index=2,
    containing_service=None,
    input_type=_GETMODELDATADIFFREQUEST,
    output_type=_GETMODELDATADIFFREQUEST_RESPONSE,
    serialized_options=b'\202\323\344\223\002\033\022\031/v1/data/getModelDataDiff',
  ),
])
_sym_db.RegisterServiceDescriptor(_MODELDATASERVICE)

DESCRIPTOR.services_by_name['ModelDataService'] = _MODELDATASERVICE

# @@protoc_insertion_point(module_scope)
