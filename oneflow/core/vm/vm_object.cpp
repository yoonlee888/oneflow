/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/vm/vm_object.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void DependenceAccess::__Init__() {
  clear_instruction();
  clear_mirrored_object();
}

void DependenceAccess::__Init__(Instruction* instruction, MirroredObject* mirrored_object,
                                OperandAccessType access_type) {
  __Init__();
  set_instruction(instruction);
  set_mirrored_object(mirrored_object);
  set_access_type(access_type);
}

}  // namespace vm
}  // namespace oneflow
