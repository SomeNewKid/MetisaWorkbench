---
name: robot-storage-inspection
description: Use when asked whether a robot is ready to be stored.
---

Read `references/checklist.md` to determine the checks to be made
before determining whether a robot is ready to be stored.

Execute the `scripts/storage_available.star` Starlark script
to determine if a storage bay is available for storing the robot,
using script arguments of `{"robot_name": robot_name}`.