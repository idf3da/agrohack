# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: resume.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='resume.proto',
  package='pb',
  syntax='proto3',
  serialized_pb=_b('\n\x0cresume.proto\x12\x02pb\"\xd9\x02\n\x06Resume\x12\n\n\x02ID\x18\x01 \x01(\t\x12\x10\n\x08Location\x18\x02 \x01(\t\x12\r\n\x05Title\x18\x03 \x01(\t\x12\r\n\x05Specs\x18\x04 \x01(\t\x12\x0e\n\x06Salary\x18\x05 \x01(\t\x12\x0b\n\x03Zgr\x18\x06 \x03(\t\x12\x10\n\x08\x45xpTitle\x18\x07 \x01(\t\x12\x12\n\nExperience\x18\x08 \x03(\t\x12\x0e\n\x06Skills\x18\t \x03(\t\x12\x0e\n\x06\x44river\x18\n \x01(\t\x12\r\n\x05\x41\x62out\x18\x0b \x01(\t\x12\x10\n\x08Recomend\x18\x0c \x01(\t\x12\x11\n\tPortfolio\x18\r \x01(\t\x12\x11\n\tEducation\x18\x0e \x03(\t\x12\r\n\x05Langs\x18\x0f \x01(\t\x12\x1c\n\x14\x41\x64\x64itional_education\x18\x10 \x03(\t\x12\r\n\x05Tests\x18\x11 \x03(\t\x12\x14\n\x0c\x43\x65rtificates\x18\x12 \x01(\t\x12\x17\n\x0f\x41\x64\x64itional_info\x18\x13 \x01(\t\"\x1b\n\rResumeRequest\x12\n\n\x02ID\x18\x01 \x03(\t\"-\n\x0eResumeResponse\x12\x1b\n\x07resumes\x18\x01 \x03(\x0b\x32\n.pb.Resume2~\n\rResumeService\x12\x35\n\nGetResumes\x12\x11.pb.ResumeRequest\x1a\x12.pb.ResumeResponse\"\x00\x12\x36\n\x0bSendResumes\x12\x12.pb.ResumeResponse\x1a\x11.pb.ResumeRequest\"\x00\x42\x06Z\x04./pbb\x06proto3')
)




_RESUME = _descriptor.Descriptor(
  name='Resume',
  full_name='pb.Resume',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ID', full_name='pb.Resume.ID', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Location', full_name='pb.Resume.Location', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Title', full_name='pb.Resume.Title', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Specs', full_name='pb.Resume.Specs', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Salary', full_name='pb.Resume.Salary', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Zgr', full_name='pb.Resume.Zgr', index=5,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ExpTitle', full_name='pb.Resume.ExpTitle', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Experience', full_name='pb.Resume.Experience', index=7,
      number=8, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Skills', full_name='pb.Resume.Skills', index=8,
      number=9, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Driver', full_name='pb.Resume.Driver', index=9,
      number=10, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='About', full_name='pb.Resume.About', index=10,
      number=11, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Recomend', full_name='pb.Resume.Recomend', index=11,
      number=12, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Portfolio', full_name='pb.Resume.Portfolio', index=12,
      number=13, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Education', full_name='pb.Resume.Education', index=13,
      number=14, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Langs', full_name='pb.Resume.Langs', index=14,
      number=15, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Additional_education', full_name='pb.Resume.Additional_education', index=15,
      number=16, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Tests', full_name='pb.Resume.Tests', index=16,
      number=17, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Certificates', full_name='pb.Resume.Certificates', index=17,
      number=18, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='Additional_info', full_name='pb.Resume.Additional_info', index=18,
      number=19, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=366,
)


_RESUMEREQUEST = _descriptor.Descriptor(
  name='ResumeRequest',
  full_name='pb.ResumeRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ID', full_name='pb.ResumeRequest.ID', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=368,
  serialized_end=395,
)


_RESUMERESPONSE = _descriptor.Descriptor(
  name='ResumeResponse',
  full_name='pb.ResumeResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='resumes', full_name='pb.ResumeResponse.resumes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=397,
  serialized_end=442,
)

_RESUMERESPONSE.fields_by_name['resumes'].message_type = _RESUME
DESCRIPTOR.message_types_by_name['Resume'] = _RESUME
DESCRIPTOR.message_types_by_name['ResumeRequest'] = _RESUMEREQUEST
DESCRIPTOR.message_types_by_name['ResumeResponse'] = _RESUMERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Resume = _reflection.GeneratedProtocolMessageType('Resume', (_message.Message,), dict(
  DESCRIPTOR = _RESUME,
  __module__ = 'resume_pb2'
  # @@protoc_insertion_point(class_scope:pb.Resume)
  ))
_sym_db.RegisterMessage(Resume)

ResumeRequest = _reflection.GeneratedProtocolMessageType('ResumeRequest', (_message.Message,), dict(
  DESCRIPTOR = _RESUMEREQUEST,
  __module__ = 'resume_pb2'
  # @@protoc_insertion_point(class_scope:pb.ResumeRequest)
  ))
_sym_db.RegisterMessage(ResumeRequest)

ResumeResponse = _reflection.GeneratedProtocolMessageType('ResumeResponse', (_message.Message,), dict(
  DESCRIPTOR = _RESUMERESPONSE,
  __module__ = 'resume_pb2'
  # @@protoc_insertion_point(class_scope:pb.ResumeResponse)
  ))
_sym_db.RegisterMessage(ResumeResponse)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('Z\004./pb'))

_RESUMESERVICE = _descriptor.ServiceDescriptor(
  name='ResumeService',
  full_name='pb.ResumeService',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=444,
  serialized_end=570,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetResumes',
    full_name='pb.ResumeService.GetResumes',
    index=0,
    containing_service=None,
    input_type=_RESUMEREQUEST,
    output_type=_RESUMERESPONSE,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='SendResumes',
    full_name='pb.ResumeService.SendResumes',
    index=1,
    containing_service=None,
    input_type=_RESUMERESPONSE,
    output_type=_RESUMEREQUEST,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_RESUMESERVICE)

DESCRIPTOR.services_by_name['ResumeService'] = _RESUMESERVICE

# @@protoc_insertion_point(module_scope)