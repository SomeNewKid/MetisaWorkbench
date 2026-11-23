---
name: robot-storage-inspection
description: Use when asked whether a robot is ready to be stored.
---

Read `references/checklist.md` to determine the checks to be made
before determining whether a robot is ready to be stored.

Execute the `scripts/storage_available.star` Starlark script
using the `execute_skill_script` tool
to determine if a storage bay is available for storing the robot,
using script parameters of `{"robot_name": robot_name}`.